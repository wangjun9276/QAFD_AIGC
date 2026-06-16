"""IQAG model definitions used by the test entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .clip_gemdwt import clip
from .loralib.utils import apply_lora


class IQAGBackbone(nn.Module):
    """CLIP-based IQAG backbone compatible with existing GEMCLIP2 checkpoints."""

    def __init__(
        self,
        num_classes: int,
        clip_checkpoint: str,
        *,
        ce: bool = True,
        concat: bool = False,
        cross: bool = True,
        head: int = 8,
        depth: int = 4,
        args: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        model_args: Dict[str, object] = dict(args or {})
        model_args.setdefault("num_vit_adapter", depth)
        model_args.setdefault("lora", False)

        self.clipmodel, _ = clip.load(
            clip_checkpoint,
            device="cpu",
            jit=False,
            args=model_args,
        )
        self.clipmodel.float()
        self.ctx_dim = self.clipmodel.ln_final.weight.shape[0]
        self.concat = concat
        self.cross = cross

        self.context_length = self.clipmodel.context_length
        self.visual = self.clipmodel.visual
        self.transformer = self.clipmodel.transformer
        self.vocab_size = self.clipmodel.vocab_size
        self.token_embedding = self.clipmodel.token_embedding
        self.positional_embedding = self.clipmodel.positional_embedding
        self.ln_final = self.clipmodel.ln_final
        self.text_projection = self.clipmodel.text_projection
        self.logit_scale = self.clipmodel.logit_scale
        self.dtype = self.visual.conv1.weight.dtype

        if ce:
            self.classification_head = nn.Linear(self.ctx_dim, num_classes)
        else:
            self.classification_head = nn.Sequential(nn.Linear(self.ctx_dim, 512))

        transformer_width = self.ctx_dim
        # Retained for compatibility with existing checkpoints.
        self.liqetoken_embedding = nn.Embedding(self.vocab_size, transformer_width)

        if self.concat:
            self.fusionlayer = nn.Linear(transformer_width * 2, transformer_width)
            self.fusionnorm = nn.LayerNorm(transformer_width)
        elif self.cross:
            # Retained for checkpoint compatibility. The released IQAG path uses
            # additive text fusion, matching the original test implementation.
            self.fusionlayer1 = nn.MultiheadAttention(transformer_width, head)
            self.fusionnorm1 = nn.LayerNorm(transformer_width)
            self.fusionlinear1 = nn.Linear(transformer_width, transformer_width)
            self.activation = nn.ReLU()
            self.fusiondropout = nn.Dropout(0.1)
            self.fusionnorm2 = nn.LayerNorm(transformer_width)

        self.depth = depth
        if bool(model_args["lora"]):
            apply_lora(model_args, self.clipmodel)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        _, outputs = self.transformer(x)
        _, x, _ = outputs
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        indices = torch.arange(x.shape[0], device=x.device)
        return x[indices, tokens.argmax(dim=-1)] @ self.text_projection

    def text_fusion(
        self,
        class_tokens: torch.Tensor,
        iqa_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        class_features = self.encode_text(class_tokens)
        iqa_features = self.encode_text(iqa_tokens)
        return class_features + iqa_features, iqa_features

    def encode_image(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.visual(
            [images.type(self.dtype), text_features.type(self.dtype)],
            return_full=False,
        )[1]

    def forward(
        self,
        images: torch.Tensor,
        class_tokens: torch.Tensor,
        iqa_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_features, _ = self.text_fusion(class_tokens, iqa_tokens)
        image_features = self.encode_image(images, text_features)
        logits = self.classification_head(image_features)
        return image_features, text_features, logits


class IQAGInferenceModel(nn.Module):
    """Run the real and fake prompt branches and return two score spaces."""

    def __init__(self, backbone: IQAGBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _paired_similarity(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        return logit_scale * (image_features * text_features).sum(dim=1, keepdim=True)

    def forward(
        self,
        images: torch.Tensor,
        class_tokens: torch.Tensor,
        iqa_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_tokens = class_tokens[:, 0, :]
        fake_tokens = class_tokens[:, 1, :]

        real_image, real_text, classification_logits = self.backbone(
            images, real_tokens, iqa_tokens
        )
        fake_image, fake_text, _ = self.backbone(images, fake_tokens, iqa_tokens)

        scale = self.backbone.logit_scale.exp()
        real_score = self._paired_similarity(real_image, real_text, scale)
        fake_score = self._paired_similarity(fake_image, fake_text, scale)
        similarity_logits = torch.cat((real_score, fake_score), dim=1)
        return similarity_logits, classification_logits


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> Tuple[list[str], list[str]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state_dict mapping.")

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key
        for prefix in ("module.", "backbone.", "clsmodel."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


# Backward-compatible name used in the original project.
GEMCLIP2 = IQAGBackbone
