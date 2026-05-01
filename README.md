# Fake3DGS

This is the official PyTorch implementation of the paper *"[Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering](https://arxiv.org/abs/2604.27590)"* (ICPR 2026).

<img src="imgs/renderings.png" alt="" style="zoom: 60%;" />

## 1. Setup

### Environment Setup 

You can find all the packages and dependencies in the environment.yml file. 
If you have conda, you can simply run

```
conda env create -f environment.yml

```

## 2. Download the dataset

Download the dataset:
[original_samples](https://ailb-web.ing.unimore.it/publicfiles/gbData/Fake3DGS/dataset/original_compressed.zip) | [edited_samples](https://ailb-web.ing.unimore.it/publicfiles/gbData/Fake3DGS/dataset/fake_compressed.zip)

The dataset contains nerfstudio checkpoints compressed. 
You need to preprocess the data with extract_data.py file to extract each gaussian splats features inside a pkl file, according to our experimentation. 
  
```
python extract_data.py --input_dir root/Fake3DGS/original_compressed/ --output_dir gaussian_pickels/real --label 0 #1 if fake

```


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

To test the model, download the corresponding pretrained weights [HERE](https://ailb-web.ing.unimore.it/publicfiles/gbData/Fake3DGS/checkpoint.pth) 
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