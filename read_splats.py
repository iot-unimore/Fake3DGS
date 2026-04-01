from PIL import Image
import numpy as np
import json
from einops import rearrange
import torch
import os

PATH = "/datasets/FG_dataset/original/1-9557-30486"


def crop_n_splats(splats, n_crop):
    
    opacities = splats["opacity"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices].squeeze(1)
    
    return splats

meta_file = os.path.join(PATH,"meta.json")
with open(meta_file) as f:
    meta = json.load(f)

def load_png(file, name):
    img = Image.open(file)
    arr = np.array(img).astype(np.float32)
    # Normalizzazione in base ai min/max dal meta
    min_val = np.array(meta[name]["mins"])
    max_val = np.array(meta[name]["maxs"])
    
    # Rescale da 0-255 a min-max
    arr = arr / 255.0 * (max_val - min_val) + min_val
    return arr


means = load_png(os.path.join(PATH,"means_l.png"), "means")  # o means_u.png se vuoi usare l'altro
scales = load_png(os.path.join(PATH,"scales.png"), "scales")
quats = load_png(os.path.join(PATH,"quats.png"), "quats")
opacities = load_png(os.path.join(PATH,"opacities.png"), "opacities")
colors = load_png(os.path.join(PATH,"sh0.png"), "sh0")

means = rearrange(torch.from_numpy(means), "H W C -> (H W) C")
scales = rearrange(torch.from_numpy(scales), "H W C -> (H W) C")
quats = rearrange(torch.from_numpy(quats), "H W C -> (H W) C")  
opacities = rearrange(torch.from_numpy(opacities).unsqueeze(-1), "H W C -> (H W) C")
colors = rearrange(torch.from_numpy(colors), "H W C -> (H W) C")


print(means.shape, scales.shape, quats.shape, opacities.shape, colors.shape)



data_dict = {
    "coord": means,
    "color": colors,
    "opacity": opacities,
    "quat": quats,
    "scale": scales
}

target_n = 65536
n_gs = means.shape[0]

if n_gs > target_n:
    n_crop = n_gs - target_n
    data_dict = crop_n_splats(data_dict, n_crop)




from transform import Collect
from functools import partial

collect = Collect(
    keys=["coord"],
    feat_keys=["color", "opacity", "quat", "scale"]
)

data = collect(data_dict)

from point_transformer import Point
data["grid_size"] = 0.01
point = Point(data)


order=("z", "z-trans")
shuffle_orders=True
point.serialization(order=order, shuffle_orders=shuffle_orders)
point.sparsify()


feat = point.feat
coord = point.coord


print(point.keys())

splat = torch.cat((coord,feat), dim=-1)



