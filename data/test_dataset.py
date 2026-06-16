"""Dataset discovery and test-time preprocessing for IQAG."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.clip_gemdwt import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_CLASS_PROMPTS = ("a real image", "an AIGC-generated image")
DEFAULT_IQA_PROMPT = "a photo with unknown visual quality and unknown processing artifacts"


def load_dataset_config(path: str | Path) -> Dict[str, Dict[str, str]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Dataset config must be a JSON object keyed by dataset name.")
    return {str(key).lower(): dict(value) for key, value in raw.items()}


def _images(path: Path, recursive: bool = False) -> List[str]:
    if not path.exists():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(str(item.resolve()) for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _collect_named_binary_dirs(base: Path) -> Tuple[List[str], List[str]]:
    real_images: List[str] = []
    fake_images: List[str] = []
    if not base.exists():
        return real_images, fake_images

    for directory in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        name = directory.name.lower()
        if name == "0_real":
            real_images.extend(_images(directory, recursive=True))
        elif name == "1_fake":
            fake_images.extend(_images(directory, recursive=True))
    return sorted(set(real_images)), sorted(set(fake_images))


def _require_path(config: Mapping[str, str], key: str, dataset: str) -> Path:
    value = config.get(key)
    if not value:
        raise KeyError(f"Dataset '{dataset}' requires '{key}' in the dataset config.")
    return Path(value).expanduser().resolve()


def collect_test_samples(
    dataset: str,
    generator: str,
    config: Mapping[str, str],
) -> Tuple[List[str], List[int]]:
    """Resolve real/fake image paths without embedding machine-specific paths."""
    name = dataset.lower()
    root = _require_path(config, "root", dataset)
    real_images: List[str] = []
    fake_images: List[str] = []

    if name == "ojha":
        real_key = "real_imagenet" if generator == "guided" else "real_laion"
        real_images = _images(_require_path(config, real_key, dataset), recursive=True)
        fake_images = _images(root / generator / "1_fake", recursive=True)
    elif name == "drct":
        real_root = Path(config.get("real_root", root / "cocoval2017")).expanduser().resolve()
        real_images = _images(real_root, recursive=True)
        fake_images = _images(root / generator / "val2017", recursive=True)
    elif name == "vlmd":
        real_root = Path(config.get("real_root", root / "Real")).expanduser().resolve()
        real_images = _images(real_root, recursive=True)
        fake_images = _images(root / generator, recursive=True)
    elif name == "dire":
        if generator in {"dalle2", "if", "midjourney", "sdv2face"}:
            domain_root = root / "celebahq"
        elif generator in {"iddpm", "ldm", "projectedgan", "sdv2bed", "stylegan"}:
            domain_root = root / "lsun_bedroom"
        else:
            domain_root = root / "imagenet"
        real_images = _images(domain_root / "real", recursive=True)
        fake_images = _images(domain_root / generator, recursive=True)
    elif name == "fakebench":
        real_images = _images(root / "real_images", recursive=True)
        fake_images = _images(root / "fake_images", recursive=True)
    elif name == "deepfake24":
        fake_images = _images(root, recursive=True)
        real_root = _require_path(config, "real_root", dataset)
        real_images = _images(real_root, recursive=True)[: len(fake_images)]
    else:
        generator_root = root / generator
        real_images, fake_images = _collect_named_binary_dirs(generator_root)

    if not real_images:
        raise FileNotFoundError(
            f"No real images found for dataset='{dataset}', generator='{generator}'. "
            f"Check the configured paths: {dict(config)}"
        )
    if not fake_images:
        raise FileNotFoundError(
            f"No fake images found for dataset='{dataset}', generator='{generator}'. "
            f"Check the configured paths: {dict(config)}"
        )

    paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    return paths, labels


def load_iqa_prompts(csv_path: Optional[str | Path]) -> Dict[str, str]:
    """Read precomputed LIQE/IQA prompts from a CSV file.

    Accepted layouts include ``image_path,label,prompt`` and files with named
    columns such as ``img_name``/``image_path`` and ``iqa_prompt``/``prompt``.
    """
    if csv_path is None:
        return {}
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IQA prompt CSV does not exist: {path}")

    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}

    header = [cell.strip().lower() for cell in rows[0]]
    path_names = {"img_name", "image_path", "path", "filename", "image"}
    prompt_names = {"iqa_prompt", "liqe_prompt", "prompt", "iqa", "description"}
    has_header = bool(set(header) & path_names) and bool(set(header) & prompt_names)

    if has_header:
        path_idx = next(i for i, value in enumerate(header) if value in path_names)
        prompt_idx = next(i for i, value in enumerate(header) if value in prompt_names)
        data_rows = rows[1:]
    else:
        path_idx, prompt_idx = 0, 2
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(path_idx, prompt_idx):
            continue
        image_path, prompt = row[path_idx].strip(), row[prompt_idx].strip()
        if image_path and prompt:
            records.append((image_path, prompt))

    basename_counts = Counter(Path(image_path).name for image_path, _ in records)
    prompts: Dict[str, str] = {}
    for image_path, prompt in records:
        prompts[str(Path(image_path).expanduser())] = prompt
        prompts[str(Path(image_path).expanduser().resolve())] = prompt
        basename = Path(image_path).name
        if basename_counts[basename] == 1:
            prompts[basename] = prompt
    return prompts


class IQAGTestDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        labels: Sequence[int],
        image_size: int = 224,
        iqa_prompts: Optional[Mapping[str, str]] = None,
        default_iqa_prompt: str = DEFAULT_IQA_PROMPT,
        class_prompts: Sequence[str] = DEFAULT_CLASS_PROMPTS,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")
        if len(class_prompts) != 2:
            raise ValueError("Exactly two class prompts are required: real and fake.")

        self.image_paths = list(image_paths)
        self.labels = [int(label) for label in labels]
        self.iqa_prompts = dict(iqa_prompts or {})
        self.default_iqa_prompt = default_iqa_prompt
        self.class_tokens = clip.tokenize(list(class_prompts))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _prompt_for(self, image_path: str) -> str:
        path = Path(image_path)
        return (
            self.iqa_prompts.get(str(path))
            or self.iqa_prompts.get(str(path.resolve()))
            or self.iqa_prompts.get(path.name)
            or self.default_iqa_prompt
        )

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        iqa_tokens = clip.tokenize([self._prompt_for(image_path)]).squeeze(0)
        return image_tensor, self.labels[index], self.class_tokens, iqa_tokens, image_path

    def __len__(self) -> int:
        return len(self.image_paths)


def build_test_loader(
    dataset: IQAGTestDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
