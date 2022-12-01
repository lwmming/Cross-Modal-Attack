# Cross-Modal-Attack
Code for the AAAI 2023 paper: "Global-Local Characteristic Excited Cross-Modal Attacks from Images to Video" (accepted).

# Environments
- python 3.7.6
- pytorch 1.9.0
- torchvision 0.10.0
- gluoncv 0.10.4.post4
- GPU: NVIDIA GeForce RTX 3090

# Data
We use datasets which are sampled from UCF-101 and Kinetics-400. You can download them from [here](https://drive.google.com/drive/folders/1O4XyLw37WqGKqFvWFaE2ps5IAD_shSpG?usp=sharing). (thanks to [@zhipeng-wei](https://github.com/zhipeng-wei)). After downloading them, you should unzip them in the current dictionary.

# Models
We use six pretrained video recognition models on UCF101 and Kinetics-400. You can download them from  [here](https://drive.google.com/drive/folders/10KOlWdi5bsV9001uL4Bn1T48m9hkgsZ2?usp=sharing) and [gluoncv](https://cv.gluon.ai/model_zoo/action_recognition.html), respectively. After downloading the UCF-101 models, you should place them into ./checkpoints.

# Training
```
python main_k400.py/main_ucf.py --batchsize 1 --gpu 0 --adv_path your_adv_path
```
Then generated video adversarial examples will be saved in your_adv_path in 'npy' format.

# Testing
```
python test_k400.py/test_ucf.py --gpu 0 --adv_path your_adv_path
```
The ASRs on six video models will be presented in ./your_adv_path/top1_asr_all_models.json.
