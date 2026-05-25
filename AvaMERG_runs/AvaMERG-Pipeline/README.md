# AvaMERG-Pipeline
This is the baseline code for [**The ACM Multimedia 2025 Grand Challenge of Avatar-based Multimodal Empathetic Conversation**](https://avamerg.github.io/MM25-challenge/)

## 📰 News

- **2025-04-27**: 🎉 We released the **Task 1** baseline code!

## TODO
- [x] Release baseline code for **Task 2**.  
(The following two projects can be referred to for the implementation of speech and talking-face video generation:
[StyleTTS2](https://github.com/yl4579/StyleTTS2) and [Dreamtalk](https://github.com/ali-vilab/dreamtalk))

## Environment
```
    conda create -n merg python=3.8.0
    conda activate merg
    pip install -r requirements.txt
```
## Pretrained Model Checkpoints Download
- ImageBind: The pre-trained checkpoint can be downloaded from [ImageBind](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) with version huge. Afterward, put the imagebind_huge.pth file at [ckpt/pretrained_ckpt/imagebind_ckpt/huge].
- Vicuna: Prepare the weight files as instructed in the [prepare_vicuna.md](ckpt/pretrained_ckpt/prepare_vicuna.md). Then put the pre-trained model at [ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0].
(Alternatively, other backbone models may also be selected.)

## Download the Training Data
The training audio and video data should be downloaded from [Hugging Face](https://huggingface.co/datasets/ZhangHanXD/AvaMERG) and placed in the [train](merg_data/train) directory.

## Training
- Run the training script using:
        ```
        ./scripts/train.sh
        ```
