"""Dataset registry used by the IQAG evaluation entry point."""

from __future__ import annotations

from typing import Dict, List


DEFAULT_GENERATORS: Dict[str, List[str]] = {
    "cnnspot": [
        "biggan", "cyclegan", "crn", "gaugan", "deepfake", "imle",
        "progan", "san", "seeingdark", "stargan", "stylegan",
        "stylegan2", "whichfaceisreal",
    ],
    "genimage": [
        "ADM", "DALLE2", "Glide", "Midjourney", "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5", "VQDM", "wukong",
    ],
    "tan": [
        "AttGAN", "BEGAN", "CramerGAN", "InfoMaxGAN", "MMDGAN",
        "RelGAN", "S3GAN", "SNGAN", "STGAN",
    ],
    "ojha": [
        "guided", "glide_100_10", "glide_100_27", "glide_50_27",
        "dalle", "ldm_100", "ldm_200", "ldm_200_cfg",
    ],
    "drct": [
        "lcm-lora-sdv1-5", "ldm-text2im-large-256", "sd-controlnet-canny",
        "sdxl-turbo", "stable-diffusion-xl-refiner-1.0",
        "controlnet-canny-sdxl-1.0", "lcm-lora-sdxl",
        "sd21-controlnet-canny", "sd-turbo", "stable-diffusion-xl-base-1.0",
    ],
    "vlmd": [
        "ADM", "DDPM", "Diff-ProjectedGAN", "Diff-StyleGAN2", "IDDPM",
        "LDM", "PNDM", "ProjectedGAN", "StyleGAN",
    ],
    "dire": [
        "dalle2", "if", "midjourney", "sdv2face", "adm", "sdv1",
        "iddpm", "ldm", "projectedgan", "sdv2bed", "stylegan",
    ],
    "wildrf": ["facebook", "reddit", "twitter"],
    "papers": ["VARsC"],
    "twinsynths": ["TwinSynths_GAN", "TwinSynths_DM"],
    "fakebench": ["test"],
    "deepfake24": ["image"],
}


def get_generators(dataset: str) -> List[str]:
    """Return the default generator list for a dataset."""
    key = dataset.lower()
    for registered_name, generators in DEFAULT_GENERATORS.items():
        if registered_name.lower() == key:
            return list(generators)
    raise KeyError(
        f"No default generator list is registered for '{dataset}'. "
        "Pass --generators explicitly."
    )
