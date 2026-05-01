import os
import gc
import joblib
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Process Gaussian data")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input folder (compressed)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output folder",
        default="./gaussian_pickles"
    )

    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        required=True,
        help="0 = Real, 1 = Fake"
    )

    parser.add_argument(
        "--crop_to",
        type=int,
        default=None,
        help="Crop dimension (e.g., 65536). Default: None"
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for processing scenes. Default: 0"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    CONFIG = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "label": args.label,
        "crop_to": args.crop_to,
        "start_index": args.start_index
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