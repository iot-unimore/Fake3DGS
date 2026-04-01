import torch
from functools import partial
from dataset import DefaultDataset
from dataloader import point_collate_fn
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import argparse
from point_transformer import PointTransformerV3
import pandas as pd

print(f"Device: {torch.cuda.get_device_name(0)}")


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




parser = argparse.ArgumentParser()
parser.add_argument("-bs","--batch_size", type=int, default=2)
parser.add_argument("-e","--epochs", type=int, default=10)
parser.add_argument("-r", "--resume", type=str, default=None, help="Path del checkpoint")
parser.add_argument("-c", "--config", type=str, default="splits/mixed_training.pkl", help="Experiment type")
parser.add_argument("-n", "--name", type=str, default=None, help="Name of the run")
parser.add_argument("-a", "--ablative", type=str, default=None )





args = parser.parse_args()

if args.ablative is not None:
    features = ablations[args.ablative]["feat"]
    num_feat = ablations[args.ablative]["num_feat"]
else:
    features = ("opacity", "scale", "quat", "s0", "sh")
    num_feat = 56




bs = args.batch_size
print(f"Running  {args.name}")
# set seed
manualSeed = 1234
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logs = {"epoch" : [], "loss" :[], "accuracy":[]}

config = {
    "learning_rate": 1e-4,
    "architecture": "PointTransformerv3+MLP",
    "dataset": "Custom",
    "epochs": args.epochs,
    "batch_size": bs
}

transform_train=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomShift", shift=[[-0.2, 0.2], [-0.2, 0.2], [0, 0]]),
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "lang_feat",
                    "valid_feat_mask",
                    "pc_coord",
                    "pc_segment",
                    "label"
                ),
                feat_keys=features
            ),
        ]
    
transform_test =[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "lang_feat",
                    "valid_feat_mask",
                    "pc_coord",
                    "pc_segment",
                    "label"
                ),
                feat_keys=features
            ),
        ]


train_data = DefaultDataset(split="train", transform=transform_train, configuration=args.config)
test_data = DefaultDataset(split="test", transform=transform_test, configuration=args.config)

train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=bs,
            shuffle=True,
            num_workers=8,
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True

        )



order=("z", "z-trans")
shuffle_orders=True

"""
model = PointTransformerV3(
    in_channels=56,
    enable_flash=False,
    order=order,
    shuffle_orders=shuffle_orders,
    enc_depths=(2,2,2,4,2),
    enc_channels=(32,64,128,256,384),
    enc_num_head=(2,4,8,8,12),
    enc_patch_size=(256,256,256,256,256),
    dec_depths=(2,2,2,2),
    dec_channels=(64,128,128,256),
    dec_num_head=(4,4,8,8),
    dec_patch_size=(256,256,256,256),
    cls_mode=True
)
model = model.cuda()
"""
model = PointTransformerV3(
    in_channels=num_feat,
    enable_flash=False,
    order=("z", "z-trans"),
    shuffle_orders=True,

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

criterion = nn.CrossEntropyLoss()

lr = 1e-4
if args.resume is not None:
    checkpoint = torch.load(f"checkpoints/{args.resume}")
    model.load_state_dict(checkpoint['model'])
    lr = checkpoint['lr']
    logs = pd.read_csv(f"{args.name}.csv").to_dict(orient="list")

optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=1e-4)

if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])



device = "cuda:0"

epochs = args.epochs

if args.resume:
    start = checkpoint['epoch']+1
else:
    start = 0

for e in range(start, epochs):

        progress_bar = tqdm(total=len(train_loader))
        it = 0
        
        model.train()

        total_loss = 0
        for data in train_loader:

            optimizer.zero_grad()
            data["grid_size"]= torch.Tensor([0.01])
            data = to_device(data, device="cuda:0")

            label = data["label"].long()           

            
            out = model(data)

            loss = criterion(out, label)
            
            it += 1

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": total_loss/it})
            
        

        torch.save({'epoch': e,
                    'lr': lr,
                    'optimizer': optimizer.state_dict(),
            'model':model.state_dict()}, f"checkpoints/{args.name}_{e}.pth")
        
        train_loss = total_loss/it


        progress_bar = tqdm(total=len(test_loader))

        model.eval()

        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                
                data["grid_size"]= torch.Tensor([0.01])
                data = to_device(data, device="cuda:0")
                label = data['label'].long()

                out = model(data)

                out = torch.argmax(out,dim=-1)
                correct += (out==label).sum().item()
                total += label.shape[0]


                progress_bar.update(1)
                progress_bar.set_postfix({"Test Acc": correct/total})
                

        test_acc = correct/total
        
        logs["epoch"].append(e + 1)
        logs["loss"].append(train_loss)
        logs["accuracy"].append(test_acc)
        df = pd.DataFrame(logs)
        df.to_csv(f"{args.name}.csv", index=False)

            
        print("Final Test Acc:", correct/total)