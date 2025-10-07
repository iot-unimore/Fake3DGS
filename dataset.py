import os
import glob
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
import joblib
import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from transform import Compose, TRANSFORMS



class DefaultDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root="gaussian_pickles/",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        sample_tail_classes=False,
        filtered_scene=None,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.sample_tail = sample_tail_classes

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list(filtered_scene=filtered_scene)
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self, filtered_scene=None):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        else:
            raise NotImplementedError
        if filtered_scene is not None:
            data_list = [
                d
                for d in data_list
                if os.path.basename(d).split("_")[0] not in filtered_scene
            ]
        return data_list

    def get_data(self, idx):
        #data_path = self.data_list[idx % len(self.data_list)]
        data_path = os.path.join(self.data_root, self.split)
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}

        if "fake" in name:
            label = 1
        else:   
            label = 0

        
        
        data = joblib.load(os.path.join(data_path, name))
    
        
        
        #opacity = data["opacity"]

        # ordina per opacità decrescente
        #values, indices = np.sort(opacity)[::-1]

        # prendi i primi 10k indici (se N >= 10000)
        #topk_indices = indices[:10000]

        data_dict = {
        "coord": data["coord"],
        "color": data["color"],
        "opacity": data["opacity"],
        "quat": data["quat"],
        "scale": data["scale"],
        "label": label,
        "name": name
    }

    
    
        """
            data_dict = {
            "coord": data["coord"][topk_indices],
            "color": data["color"][topk_indices],
            "opacity": data["opacity"][topk_indices],
            "quat": data["quat"][topk_indices],
            "scale": data["scale"][topk_indices],
            "label": label,
            "name": name
        }
        """

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        label = data_dict["label"]
        data_dict = self.transform(data_dict)
        data_dict["label"] = label
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segment=data_dict.pop("segment", None), name=data_dict.pop("name", None)
        )
        if "coord" in data_dict:
            result_dict["coord"] = data_dict["coord"]  # needed by ZeroShotSemSegTester
        if "pc_coord" in data_dict:
            result_dict["pc_coord"] = data_dict["pc_coord"]
        if "pc_segment" in data_dict:
            result_dict["pc_segment"] = data_dict["pc_segment"]
        if "origin_coord" in data_dict:
            result_dict["origin_coord"] = data_dict.pop("origin_coord")
        if "origin_feat_mask" in data_dict:
            result_dict["origin_feat_mask"] = data_dict.pop("origin_feat_mask")
        if "origin_instance" in data_dict:
            result_dict["origin_instance"] = data_dict.pop("origin_instance")
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                aug(deepcopy(data_dict))
            )  # this comsumes a lot of memory

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


