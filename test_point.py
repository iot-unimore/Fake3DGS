import torch
from functools import partial
from dataset import DefaultDataset
from dataloader import point_collate_fn
import torch.nn as nn
from tqdm import tqdm
import argparse
from point_transformer import PointTransformerV3
import random
import numpy as np

print(f"Device: {torch.cuda.get_device_name(0)}")



parser = argparse.ArgumentParser()
parser.add_argument("-bs","--batch_size", type=int, default=2)
parser.add_argument("-e","--epochs", type=int, default=10)
parser.add_argument("-ckpt","--ckpt", type=str, required=True)
parser.add_argument("-t","--test_set", type=str, default="mixed_training.pkl")
parser.add_argument("-a", "--ablative", type=str, default=None )

ablations = {
    "wo_opacity": {
        "feat": ("scale", "quat", "s0", "sh"),
        "num_feat": 55
    },
    "wo_scale": {
        "feat": ("opacity", "quat", "s0", "sh"),
        "num_feat": 53
    },
    "wo_quat": {
        "feat": ("opacity", "scale", "s0", "sh"),
        "num_feat": 52
    },
    "wo_s0": {
        "feat": ("opacity", "scale", "quat", "sh"),
        "num_feat": 53
    },
    "wo_sh": {
        "feat": ("opacity", "scale", "quat", "s0"),
        "num_feat": 11
    },
    "only_s0": {
        "feat": ("s0"),
        "num_feat": 3
    }
}




manualSeed = 1234
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

args = parser.parse_args()
if args.ablative is not None:
    features = ablations[args.ablative]["feat"]
    num_feat = ablations[args.ablative]["num_feat"]
else:
    features = ("opacity", "scale", "quat", "s0", "sh")
    num_feat = 56


bs = args.batch_size

project="fake_gaussian"
config = {
    "learning_rate": 1e-4,
    "architecture": "PointTransformerv3+MLP",
    "dataset": "Custom",
    "epochs": args.epochs,
    "batch_size": bs
}

transform_test=[
    dict(type="NormalizeCoord"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord","grid_coord","segment","lang_feat","valid_feat_mask","pc_coord","pc_segment","label"),
        feat_keys=features,
    ),
]


test_data = DefaultDataset(split="test", transform=transform_test, configuration=args.test_set)

print(args.test_set)
print(args.ckpt)

test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )




model = PointTransformerV3(
    in_channels=num_feat,
    enable_flash=False,
    order=("z", "z-trans"),
    shuffle_orders=False,

    enc_depths=(1, 1, 2, 2),
    enc_channels=(32, 64, 128, 128),
    enc_num_head=(2, 4, 4, 4),
    enc_patch_size=(32, 32, 32, 32),

    stride=(1, 2, 2), 

    dec_depths=(1, 1),
    dec_channels=(64, 128),
    dec_num_head=(2, 4),
    dec_patch_size=(16, 16),

    cls_mode=True
).cuda()

from transform import to_device

model.load_state_dict(torch.load(args.ckpt)['model'])


device = "cuda:0"

epochs = args.epochs



progress_bar = tqdm(total=len(test_loader))

model.eval()

TP = 0
TN = 0
FP = 0
FN = 0

total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        
        data["grid_size"]= torch.Tensor([0.01])
        data = to_device(data, device="cuda:0")
        label = data['label'].long()

        pred = model(data)
        pred = torch.argmax(pred,dim=-1)


        TP += ((label == 1) & (pred == 1)).sum().item()
        TN += ((label == 0) & (pred == 0)).sum().item()
        FP += ((label == 0) & (pred == 1)).sum().item()
        FN += ((label == 1) & (pred == 0)).sum().item()

        correct += (pred==label).sum().item()
        total += label.shape[0]


        progress_bar.update(1)
        progress_bar.set_postfix({"Test Acc": correct/total})
        

print(f"Final Test Acc: {(correct/total)*100:.1f}%")
print(f"TP rate: {(TP / (TP + FN))*100:.1f}%")
print(f"TN rate: {(TN / (TN + FP))*100:.1f}%")
