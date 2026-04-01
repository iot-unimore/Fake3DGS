# Fake3DGS

This is the official PyTorch implementation of the paper *"[Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering](https://arxiv.org/)"* (ICPR 2026).

<img src="imgs/renderings.png" alt="" style="zoom: 60%;" />

## 1. Setup

### Environment Setup 

You can find all the packages and dependencies in the environment.yml file. 
If you have conda, you can simply run

```
conda env create -f environment.yml

```

## 2. Download the dataset

Download the dataset [HERE]()
There are two versions of the dataset:
- The first one contains nerfstudio checkpoints compressed.
- The second one contains each gaussian splats extracted features inside a pkl file. 
  
We used the second one in model training. Make sure to extract gaussian_pickles.tar.gz inside this folder.
You can also download the first version of the dataset for further experiments. 



## 3. Training

```
python -u train.py -bs 4 -e 8 --config mixed_training.pkl --name new_mixed --resume new_mixed_5.pth
```

### Explanation of Parameters:
- `--bs` : Batch size.
- `-e` : Number of epochs.
- `--config`: config pkl file.
- `--name`: Name of the training.
- `--resume`: Checkpoint from which to resume training.

## 4. Testing 

To test the model, download the corresponding pretrained weights [HERE]() 
and place them in the folder:

```
checkpoints/
```
Then run:

```
python -u test.py -bs 4 --test_set "mixed_training.pkl" --ckpt "checkpoints/best.pth"
```

### Explanation of Parameters:
- `--bs` : Batch size.
- `--ckpt` : Model weights to load.
- `--test_set`: config pkl file.

## Citation

If you find our work useful for your project, please consider citing the paper:

```bibtex
@inproceedings{ddinuccifake3dgs,
  title={Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering},
  author={Di Nucci, Davide and Catalini, Riccardo and Borghi, Guido and Vezzani, Roberto},
  booktitle={Twentyeighth International Conference on Pattern Recognition},
  year=2026
}
```
---