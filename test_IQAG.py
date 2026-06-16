#!/usr/bin/env python3
"""Primary IQAG evaluation entry point.

The script evaluates classification logits, prompt similarity logits, and their
fusion over one or more AIGC benchmark datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from configs import get_generators
from data import (
    DEFAULT_CLASS_PROMPTS,
    DEFAULT_IQA_PROMPT,
    IQAGTestDataset,
    build_test_loader,
    collect_test_samples,
    load_dataset_config,
    load_iqa_prompts,
)
from models import IQAGBackbone, IQAGInferenceModel, load_checkpoint
from utils import binary_metrics, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the IQAG detector.")

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--dataset-config",
        required=True,
        help="JSON file containing dataset roots; see configs/datasets.example.json.",
    )
    data_group.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to evaluate, for example: cnnspot genimage ojha.",
    )
    data_group.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="Optional generator override. This is only valid with one dataset.",
    )
    data_group.add_argument("--image-size", type=int, default=224)
    data_group.add_argument("--batch-size", type=int, default=24)
    data_group.add_argument("--num-workers", type=int, default=8)
    data_group.add_argument(
        "--iqa-csv",
        default=None,
        help="A single precomputed IQA/LIQE prompt CSV; use with one dataset and generator.",
    )
    data_group.add_argument(
        "--iqa-csv-dir",
        default=None,
        help="Directory containing <dataset>/<generator>.csv IQA prompt files.",
    )
    data_group.add_argument("--iqa-prompt", default=DEFAULT_IQA_PROMPT)
    data_group.add_argument(
        "--class-prompts",
        nargs=2,
        metavar=("REAL_PROMPT", "FAKE_PROMPT"),
        default=DEFAULT_CLASS_PROMPTS,
    )
    data_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit per class and generator for debugging.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--checkpoint", required=True, help="Trained IQAG checkpoint.")
    model_group.add_argument(
        "--clip-checkpoint",
        default="ViT-L/14",
        help="Local CLIP checkpoint path or a registered CLIP model name.",
    )
    model_group.add_argument("--num-vit-adapter", type=int, default=4)
    model_group.add_argument("--non-strict", action="store_true")
    model_group.add_argument("--lora", action="store_true")
    model_group.add_argument(
        "--position",
        default="all",
        choices=["top", "bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
    )
    model_group.add_argument("--encoder", default="vision", choices=["text", "vision", "both"])
    model_group.add_argument("--params", nargs="+", default=["q", "k", "v"])
    model_group.add_argument("--rank", type=int, default=2)
    model_group.add_argument("--alpha", type=int, default=1)
    model_group.add_argument("--dropout-rate", type=float, default=0.25)
    model_group.add_argument("--backbone", default="ViT-L/14")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    runtime_group.add_argument("--seed", type=int, default=1234)
    runtime_group.add_argument("--fusion-weight", type=float, default=1.0)
    runtime_group.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    runtime_group.add_argument("--profile-flops", action="store_true")
    runtime_group.add_argument("--output-dir", default="results/iqag_test")

    args = parser.parse_args()
    if args.generators is not None and len(args.datasets) != 1:
        parser.error("--generators can only be used when exactly one dataset is selected.")
    if args.iqa_csv is not None and (
        len(args.datasets) != 1 or args.generators is None or len(args.generators) != 1
    ):
        parser.error("--iqa-csv requires one dataset and one explicit generator.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")
    return args


def _model_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "num_vit_adapter": args.num_vit_adapter,
        "lora": args.lora,
        "position": args.position,
        "encoder": args.encoder,
        "params": args.params,
        "r": args.rank,
        "alpha": args.alpha,
        "dropout_rate": args.dropout_rate,
        "backbone": args.backbone,
    }


def _resolve_iqa_csv(
    args: argparse.Namespace,
    dataset: str,
    generator: str,
) -> Path | None:
    if args.iqa_csv:
        return Path(args.iqa_csv)
    if not args.iqa_csv_dir:
        return None

    root = Path(args.iqa_csv_dir).expanduser()
    candidates = [
        root / dataset / f"{generator}.csv",
        root / dataset.lower() / f"{generator}.csv",
        root / f"{generator}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _profile_model(
    model: IQAGInferenceModel,
    image_size: int,
    device: torch.device,
) -> None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError("Install fvcore to use --profile-flops.") from error

    images = torch.randn(1, 3, image_size, image_size, device=device)
    class_tokens = torch.zeros(1, 2, 77, dtype=torch.int32, device=device)
    iqa_tokens = torch.zeros(1, 77, dtype=torch.int32, device=device)
    analysis = FlopCountAnalysis(model, (images, class_tokens, iqa_tokens))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameters / 1e6:.2f} M")
    print(f"Model FLOPs: {analysis.total() / 1e9:.2f} G")


def _evaluate_generator(
    model: IQAGInferenceModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fusion_weight: float,
    amp: bool,
    csv_path: Path,
) -> Dict[str, Dict[str, float]]:
    labels_all: List[np.ndarray] = []
    probabilities: Dict[str, List[np.ndarray]] = {
        "classification": [],
        "similarity": [],
        "fusion": [],
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "label",
                "classification_prob_real",
                "classification_prob_fake",
                "similarity_prob_real",
                "similarity_prob_fake",
                "fusion_prob_real",
                "fusion_prob_fake",
            ]
        )

        progress = tqdm(loader, desc=csv_path.stem, leave=False)
        with torch.inference_mode():
            for images, labels, class_tokens, iqa_tokens, image_paths in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                class_tokens = class_tokens.to(device, non_blocking=True)
                iqa_tokens = iqa_tokens.to(device, non_blocking=True)

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp and device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    similarity_logits, classification_logits = model(
                        images, class_tokens, iqa_tokens
                    )
                    fusion_logits = classification_logits + fusion_weight * similarity_logits

                classification_prob = torch.softmax(classification_logits, dim=1)
                similarity_prob = torch.softmax(similarity_logits, dim=1)
                fusion_prob = torch.softmax(fusion_logits, dim=1)

                labels_cpu = labels.cpu().numpy()
                classification_cpu = classification_prob.float().cpu().numpy()
                similarity_cpu = similarity_prob.float().cpu().numpy()
                fusion_cpu = fusion_prob.float().cpu().numpy()

                labels_all.append(labels_cpu)
                probabilities["classification"].append(classification_cpu[:, 1])
                probabilities["similarity"].append(similarity_cpu[:, 1])
                probabilities["fusion"].append(fusion_cpu[:, 1])

                for index, image_path in enumerate(image_paths):
                    writer.writerow(
                        [
                            image_path,
                            int(labels_cpu[index]),
                            *classification_cpu[index].tolist(),
                            *similarity_cpu[index].tolist(),
                            *fusion_cpu[index].tolist(),
                        ]
                    )

    labels = np.concatenate(labels_all)
    return {
        name: binary_metrics(labels, np.concatenate(score_chunks))
        for name, score_chunks in probabilities.items()
    }


def _mean_metrics(
    generator_results: Sequence[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for score_type in ("classification", "similarity", "fusion"):
        output[score_type] = {}
        for metric in ("acc", "auc", "ap", "real_acc", "fake_acc"):
            values = np.asarray(
                [result[score_type][metric] for result in generator_results],
                dtype=np.float64,
            )
            output[score_type][metric] = float(np.nanmean(values))
    return output


def _print_metrics(prefix: str, metrics: Mapping[str, Mapping[str, float]]) -> None:
    for score_type, values in metrics.items():
        print(
            f"{prefix} | {score_type:14s} "
            f"ACC={values['acc']:.4f} AUC={values['auc']:.4f} AP={values['ap']:.4f} "
            f"RealACC={values['real_acc']:.4f} FakeACC={values['fake_acc']:.4f}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    dataset_config = load_dataset_config(args.dataset_config)
    backbone = IQAGBackbone(
        num_classes=2,
        clip_checkpoint=args.clip_checkpoint,
        ce=True,
        concat=False,
        cross=True,
        depth=args.num_vit_adapter,
        args=_model_args(args),
    )
    missing, unexpected = load_checkpoint(
        backbone,
        args.checkpoint,
        strict=not args.non_strict,
    )
    if missing or unexpected:
        print(f"Checkpoint missing keys: {missing}")
        print(f"Checkpoint unexpected keys: {unexpected}")

    model = IQAGInferenceModel(backbone).to(device).eval()
    if args.profile_flops:
        _profile_model(model, args.image_size, device)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    complete_summary: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "clip_checkpoint": args.clip_checkpoint,
        "fusion_weight": args.fusion_weight,
        "datasets": {},
    }
    summary_rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        config_key = dataset.lower()
        if config_key not in dataset_config:
            raise KeyError(
                f"Dataset '{dataset}' is missing from {args.dataset_config}."
            )
        generators = args.generators or get_generators(dataset)
        dataset_results: Dict[str, object] = {}
        generator_metric_list: List[Mapping[str, Mapping[str, float]]] = []

        print(f"\nEvaluating dataset: {dataset}")
        for generator in generators:
            image_paths, labels = collect_test_samples(
                dataset,
                generator,
                dataset_config[config_key],
            )
            if args.limit is not None:
                real_pairs = [(path, label) for path, label in zip(image_paths, labels) if label == 0][: args.limit]
                fake_pairs = [(path, label) for path, label in zip(image_paths, labels) if label == 1][: args.limit]
                selected = real_pairs + fake_pairs
                image_paths = [path for path, _ in selected]
                labels = [label for _, label in selected]

            iqa_csv = _resolve_iqa_csv(args, dataset, generator)
            iqa_prompts = load_iqa_prompts(iqa_csv)
            if iqa_csv is None:
                print(
                    f"{dataset}/{generator}: no IQA CSV found; "
                    "using the label-independent default IQA prompt."
                )

            test_dataset = IQAGTestDataset(
                image_paths=image_paths,
                labels=labels,
                image_size=args.image_size,
                iqa_prompts=iqa_prompts,
                default_iqa_prompt=args.iqa_prompt,
                class_prompts=args.class_prompts,
            )
            loader = build_test_loader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            csv_path = output_root / "predictions" / dataset / f"{generator}.csv"
            metrics = _evaluate_generator(
                model,
                loader,
                device,
                args.fusion_weight,
                args.amp,
                csv_path,
            )
            _print_metrics(f"{dataset}/{generator}", metrics)
            dataset_results[generator] = {
                "num_images": len(test_dataset),
                "iqa_csv": str(iqa_csv.resolve()) if iqa_csv else None,
                "metrics": metrics,
            }
            generator_metric_list.append(metrics)

            for score_type, values in metrics.items():
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "generator": generator,
                        "score_type": score_type,
                        **values,
                    }
                )

        mean_metrics = _mean_metrics(generator_metric_list)
        _print_metrics(f"{dataset}/MEAN", mean_metrics)
        complete_summary["datasets"][dataset] = {
            "generators": dataset_results,
            "mean": mean_metrics,
        }

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(complete_summary, handle, indent=2, ensure_ascii=False)

    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "generator",
            "score_type",
            "acc",
            "auc",
            "ap",
            "real_acc",
            "fake_acc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResults saved to: {output_root}")


if __name__ == "__main__":
    main()
