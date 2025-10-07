import torch
from functools import partial
from dataset import DefaultDataset
from dataloader import point_collate_fn
import torch.nn as nn
from tqdm import tqdm
import argparse
from point_transformer import PointTransformerV3
import wandb

print(f"Device: {torch.cuda.get_device_name(0)}")



parser = argparse.ArgumentParser()
parser.add_argument("-bs","--batch_size", type=int, default=2)
parser.add_argument("-e","--epochs", type=int, default=10)

args = parser.parse_args()

bs = args.batch_size

project="fake_gaussian"
config = {
    "learning_rate": 1e-4,
    "architecture": "PointTransformerv3+MLP",
    "dataset": "Custom",
    "epochs": args.epochs,
    "batch_size": bs
}

transform_train=[
            dict(type="CenterShift", apply_z=True),
            #dict(
            #    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            #),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            #dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            #dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            #dict(type="RandomFlip", p=0.5),
            #dict(type="RandomJitter", sigma=0.005, clip=0.01),
            #dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            #dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            #dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            #dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.015,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "segment",
                    "lang_feat",
                    "valid_feat_mask",
                ),
                return_grid_coord=True,
            ),
            #dict(type="SphereCrop", point_max=50000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="NormalizeCoord"),
            # dict(type="ShufflePoint"),
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
                ),
                feat_keys=("color", "opacity", "quat", "scale"),
            ),
        ]
    
transform_test = transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.015,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "segment",
                    "lang_feat",
                    "valid_feat_mask",
                ),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=600000, mode="random"), # spconv limitation: int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, i.e., max 698k points for inference
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
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
                ),
                feat_keys=("color", "opacity", "quat", "scale"),
            ),
        ]


train_data = DefaultDataset(split="train", transform=transform_train)
test_data = DefaultDataset(split="test", transform=transform_test)

train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=bs,
            shuffle=True,
            num_workers=8,
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )



order=("z", "z-trans")
shuffle_orders=True

model = PointTransformerV3(in_channels=11, 
                           enable_flash=False, 
                           order=order, 
                           shuffle_orders=shuffle_orders,
                           enc_depths=(2, 2, 2, 4, 2),
                           enc_channels=(32, 64, 128, 256, 384),
                           enc_num_head=(2, 4, 8, 8, 12),
                           enc_patch_size=(256, 256, 256, 256, 256),
                           dec_depths=(2, 2, 2, 2),
                           dec_channels=(64, 128, 128, 256),
                           dec_num_head=(4, 4, 8, 8),
                           dec_patch_size=(256, 256, 256, 256),
                           cls_mode=True)
model = model.cuda()

from transform import to_device

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-4, weight_decay=1e-4)


#model.load_state_dict(torch.load("point_transformer.pth"))


device = "cuda:0"

epochs = args.epochs

with wandb.init(project=project, config=config) as run:
    for e in range(epochs):

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


            """
            scenes = [] 
            value = torch.max(batch) 
            for i in range(value+1): 
                mask = (batch==i) 
                feat_i = feat[mask] 
                scenes.append(torch.mean(feat_i, dim=0, keepdim=False)) 
                scenes = torch.stack(scenes)
            """
            
            loss = criterion(out, label)
            
            it += 1

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": total_loss/it})
            
        run.log({"loss": total_loss/it})

        torch.save(model.state_dict(), f"checkpoints/{run.name[:-3]}_{e}.pth")


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
                

        run.log({"accuracy": correct/total})
                
            
        print("Final Test Acc:", correct/total)