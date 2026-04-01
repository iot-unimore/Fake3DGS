import os
import gc
import joblib
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional

from decompress import PngCompression


class SceneProcessor:
    def __init__(self, input_dir: str, output_dir: str, label: int = 0):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.label = label
        self.decompressor = PngCompression()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def crop_splats(data: Dict[str, np.ndarray], n_crop: int) -> Dict[str, np.ndarray]:
        opacities = data["opacity"].squeeze()
        keep_idx = np.argsort(opacities)[n_crop:]
        return {k: v[keep_idx] for k, v in data.items() if isinstance(v, np.ndarray)}

    @staticmethod
    def process_splats(splats: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        means = splats["means"]
        scales = torch.exp(splats["scales"])
        quats = splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True)
        opacities = torch.sigmoid(splats["opacities"])

        sh0 = splats["sh0"]
        shN = splats["shN"]
        sh_feat = torch.cat([sh0, shN], dim=1).view(-1, 16 * 3)

        return {
            "means": means.cpu().numpy(),
            "scales": scales.cpu().numpy(),
            "quats": quats.cpu().numpy(),
            "opacity": opacities.cpu().numpy(),
            "sh": sh_feat.cpu().numpy(),
        }

    def load_scene(self, scene_path: Path, crop_to: Optional[int] = None) -> None:
        try:
            splats = self.decompressor.decompress(str(scene_path))
            data = self.process_splats(splats)

            if crop_to is not None and data["means"].shape[0] > crop_to:
                n_crop = data["means"].shape[0] - crop_to
                data = self.crop_splats(data, n_crop)

            data.update({
                "label": self.label,
                "name": scene_path.name
            })

            save_path = self.output_dir / f"{scene_path.name}.pkl"
            joblib.dump(data, save_path)

        except Exception as e:
            print(f"[ERROR] {scene_path}: {e}")

        finally:
            del splats
            gc.collect()
            torch.cuda.empty_cache()

    def process_all(self, start_idx: int = 0, crop_to: Optional[int] = None):
        scenes = sorted(self.input_dir.iterdir())[start_idx:]

        for scene_path in tqdm(scenes, desc="Processing scenes"):
            self.load_scene(scene_path, crop_to=crop_to)


if __name__ == "__main__":
    CONFIG = {
        "input_dir": "/PATH/TO/original_compressed", #"/PATH/TO/fake_compressed"
        "output_dir": "./gaussian_pickles/real", #"./gaussian_pickles/fake"
        "label": 0, #0 for Real, 1 for Fake
        "crop_to": None,  # es: 65536
    }

    processor = SceneProcessor(
        input_dir=CONFIG["input_dir"],
        output_dir=CONFIG["output_dir"],
        label=CONFIG["label"]
    )

    processor.process_all(
        start_idx=CONFIG["start_index"],
        crop_to=CONFIG["crop_to"]
    )