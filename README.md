# CTNeRF: Cross-Time Transformer for Dynamic Neural Radiance Field from Monocular Video

[![arXiv](https://img.shields.io/badge/arXiv-2401.04861-b31b1b.svg)](https://arxiv.org/abs/2401.04861)

> **CTNeRF: Cross-Time Transformer for Dynamic Neural Radiance Field from Monocular Video**<br>
> [Paper(Arxiv)](https://arxiv.org/abs/2401.04861) | [Results](https://drive.google.com/file/d/10LHsemH6ImE4mghYImPtsVT1EpnbaNk5/view?usp=sharing)<br>
> Xingyu Miao, Yang Bai, Haoran Duan, Yawen Huang, Fan Wan, Yang Long, Yefeng Zheng<br>
> Accepted by Pattern Recognition (PR)



## Setup
The code is test with
* Linux 
* Anaconda 3
* Python 3.7.11
* CUDA 10.1
* 2 3090 GPU


To get started, please create the conda environment `ctnerf` by running
```
conda create --name ctnerf python=3.7
conda activate ctnerf
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install imageio scikit-image configargparse timm lpips
```
and install [COLMAP](https://colmap.github.io/install.html) manually. Then download MiDaS and RAFT weights
```
ROOT_PATH=/path/to/the/CTNeRF/folder
cd $ROOT_PATH
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/weights.zip
unzip weights.zip
rm weights.zip
```

## Dynamic Scene Dataset
The [Dynamic Scene Dataset](https://www-users.cse.umn.edu/~jsyoon/dynamic_synth/) is used to
quantitatively evaluate our method. Please download the pre-processed data by running:
```
cd $ROOT_PATH
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
unzip data.zip
rm data.zip
```

### Training
You can train a model from scratch by running:
```
cd $ROOT_PATH/
python run_nerf.py --config configs/config_Balloon2.txt
```

### Rendering from pre-trained models
We also provide pre-trained models. You can download them by running:
```
cd $ROOT_PATH/
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/logs.zip
unzip logs.zip
rm logs.zip
```

Then you can render the results directly by running:
```
python run_nerf.py --config configs/config_Balloon2.txt --render_only --ft_path $ROOT_PATH/logs/Balloon2_H270_CTNeRF_pretrain/300000.tar
```

### Evaluating

Please download the results by running:
```
cd $ROOT_PATH/
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/results.zip
unzip results.zip
rm results.zip
```

Then you can calculate the PSNR/SSIM/LPIPS by running:
```
cd $ROOT_PATH/utils
python evaluation.py
```


If you find this code useful for your research, please consider citing the following paper:

	@misc{miao2024ctnerfcrosstimetransformerdynamic,
      title={CTNeRF: Cross-Time Transformer for Dynamic Neural Radiance Field from Monocular Video}, 
      author={Xingyu Miao and Yang Bai and Haoran Duan and Yawen Huang and Fan Wan and Yang Long and Yefeng Zheng},
      year={2024},
      eprint={2401.04861},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.04861}, 
}

## Acknowledgments
Our training code is build upon
[pixelNeRF](https://github.com/sxyu/pixel-nerf),
[DynamicNeRF](https://github.com/gaochen315/DynamicNeRF), and
[NSFF](https://github.com/zl548/Neural-Scene-Flow-Fields).
Our flow prediction code is modified from [RAFT](https://github.com/princeton-vl/RAFT).
Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).
