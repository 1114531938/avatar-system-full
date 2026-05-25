# DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation [AAAI2025]
[![arXiv](https://img.shields.io/badge/arXiv-<2408.06010>-red.svg)](https://arxiv.org/abs/2408.06010)
[![ProjectPage](https://img.shields.io/badge/ProjectPage-DEEPTalk-<COLOR>.svg)](https://whwjdqls.github.io/deeptalk_website/)

<p align="center">
  <img src="./demo/teaser_final.png" alt="alt text" width="400">
</p>

Official pytorch code release of "[DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation](https://arxiv.org/abs/2408.06010)"

```
@misc{kim2024deeptalkdynamicemotionembedding,
      title={DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation}, 
      author={Jisoo Kim and Jungbin Cho and Joonho Park and Soonmin Hwang and Da Eun Kim and Geon Kim and Youngjae Yu},
      year={2024},
      eprint={2408.06010},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06010}, 
}
```
## 📨 News
🛩️ **12/Dec/24** - Released the training code

🛩️ **11/Dec/24** - Released the inference&rendering code

🛩️ **10/Dec/24** - DEEPTalk is accepted to AAAI2025

## ⚙️ Settings
❗clone this repo recursively using 
```bash
git clone --recurse-submodules <repository_url>
```
or update submodules recursively using 
```bash
git submodule update --init --recursive
```
note that spectre requires `git-lfs` which can be installed by
```bash
conda install conda-forge::git-lfs
```
### Environment
Make environment and install pytorch
```bash
conda create -n deeptalk python=3.9
conda activate deeptalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install pytorch3d and other requirements. Refer to this [page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for pytorch3d details.
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
pip install -r requirements.txt
```
- installation tip from [this issue](https://github.com/whwjdqls/DEEPTalk/issues/5)

Install osmesa and ffmpeg for headless rendering and audio, video processing. 
```bash
conda install menpo::osmesa
conda install conda-forge::ffmpeg
```
For trainig DEEPTalk on stage2, we used nvdiffrast.
Install nvdiffrast from [this repo](https://github.com/NVlabs/nvdiffrast).
```bash
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
git checkout v0.3.1
python setup.py install
```
---
### Download Checkpoints
Download DEE, FER, TH-VQVAE, DEEPTalk checkpoints from [here](https://drive.google.com/drive/u/0/folders/1vmgJCvAq96C83eU4JuUFooubL-y7Py44).
- **DEE.pt**: Place in `./DEE/checkpoint`
- **FER.pth**: Place in `./FER/checkpoint`
- **TH-VQVAE.pth**: Place in `./DEEPTalk/checkpoint/TH-VQVAE`
- **DEEPTalk.pth**: Place in `./DEEPTalk/checkpoint/DEEPTalk`

Download emotion2vec_bast.pt from the [emotion2vec repository](https://huggingface.co/emotion2vec/emotion2vec_base).
- **emotion2vec_base.pt**: Place in `./DEE/models/emo2vec/checkpoint`

Download LRS3_V_WER32.3 model from the [Spectre repository](https://github.com/filby89/spectre/blob/master/get_training_data.sh). (❗This is for Stage2 training)
- Place the LRS3_V_WER32.3 folder at `./DEEPTalk/externals/spectre/data/data/LRS3_V_WER32.3`

### Download Files
Download files from [Ringnet project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model).
- **FLAME_sample.ply**: Place in `./DEEPTalk/models/flame_models`
- **flame_dynamic_embedding.npy**: Place in `./DEEPTalk/models/flame_models`
- **flame_static_embedding.pkl**: Place in `./DEEPTalk/models/flame_models`

Download FLAME 2020 files from [FLAME website](https://flame.is.tue.mpg.de/). 
- **generic_model.pkl**: Place in `./DEEPTalk/models/flame_models`

Download head_template files from [FLAME website](https://flame.is.tue.mpg.de/). (❗This is for Stage2 training)
- related [issue](https://github.com/whwjdqls/DEEPTalk/issues/3#issuecomment-2547336926)
- **head_template.jpg**: Place in `./DEEPTalk/models/flame_models/geometry`
- **head_template.mtl**: Place in `./DEEPTalk/models/flame_models/geometry`
- **head_template.obj**: Place in `./DEEPTalk/models/flame_models/geometry`
---
## 🛹 Inference
Run the following copmmand to make a video. Results will be saved in `./DEEPTalk/outputs`.
```
cd DEEPTalk
python demo.py --audio_path {raw audio file (.wav) or sampled audio (.npy)}
```

## 📚 Dataset 
### Download Data
Download MEAD Dataset from [here](https://github.com/uniBruce/Mead).

### Process Data
Use the reconstruction method from [EMOCAV2](https://github.com/radekd91/inferno) to reconstruct FLAME parameters from MEAD.

Leave an issue if your having troubles processing MEAD. We might be able to provide the exact parameters.


## 🏋️ Training

Detailed hyperparameters of our model can be found in the supplementary details in this [link](https://drive.google.com/file/d/1skxpOMsQj84BOWuIa9-R_GjYG2DxwDQu/view?usp=sharing).
### 1. Train TH-VQVAE on MEAD FLAME parameters
Make a copy of `/DEEPTalk/checkpoint/TH-VQVAE/config_TH-VQVAE.json` and change the arguments like `data.data_dir` or `name` to train your own model.
Then run
```bash
cd DEEPTalk
python train_VQVAE.py --config {your config path}
```
### 2. Train DEEPTalk stage1
Make a copy of `/home/whwjdqls99/DEEPTalk/DEEPTalk/checkpoint/DEEPTalk/config_stage1.json` and change the arguments like `data.data_dir` or `name` to train your own model.
Then run
```bash
cd DEEPTalk
python train_DEEPTalk_stage1.py --DEEPTalk_config {your config path}
```
### 3. Train DEEPTalk stage2
Make a copy of `/home/whwjdqls99/DEEPTalk/DEEPTalk/checkpoint/DEEPTalk/config.json` and change the arguments like `data.data_dir` or `name` to train your own model.
Then run
```bash
cd DEEPTalk
python train_DEEPTalk_stage2.py --DEEPTalk_config {your config path} --checkpoint {stage1 trained model checkpoint path}
```

## Acknowledgements
We gratefully acknowledge the open-source projects that served as the foundation for our work:

- [EMOTE](https://github.com/radekd91/inferno)
- [learning2listen](https://github.com/evonneng/learning2listen)
- [PCME++](https://github.com/naver-ai/pcmepp)

## License
This code is released under the MIT License.

Please note that our project relies on various other libraries, including FLAME, PyTorch3D, and Spectre, as well as several datasets.
