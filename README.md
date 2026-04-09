# PADReg

## About
This is a Pytorch Implementation of our paper "PADReg: Physics-Aware Deformable Registration Guided by Contact Force for Ultrasound Sequences". 

PADReg is a physic-aware ultrasound registration model utilizing multi-modal data (i.e., ultrasound image data and contact force data).  
Keywords: Deformable Registration, Force Fusion, Multimodal, Ultrasound Imaging

## Dependencies
This code has been tested on Ubuntu 20.04 with a NVIDIA A100 Tensor Core GPU.

Required environments are listed in requirements.txt. 

To install the required environment, run:

```bash
pip install -r requirements.txt
```

## Usage
### **Datasets Preparation**
The registration dataset is generated from [Mus-V](https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation) dataset. 

Please put dataset in the folder "PADReg/data/"

### **Training**
```python
python PADReg/train.py -m 1 -r 0.03 -lr 0.001
```
Checkpoints will be recorded in "experiments" folder. Tensorboard is used for visualization. You can check the log file in "runs".
### **Validation**
```python
python PADReg/valid.py -c 'PADReg/' 
```
You can modify the checkpoint folder to your experiment folder. 

## Citation
```bib
@article{PADReg,
    author = {Geng, Yimeng and Zhao, Mingyang and Xu, Fan and Cao, Guanglin and Meng, Gaofeng and Liu, Hongbin},
    year = {2026},
    month = {01},
    pages = {1-8},
    title = {PADReg: Physics-Aware Deformable Registration Guided by Contact Force for Ultrasound Sequences},
    volume = {PP},
    journal = {IEEE Robotics and Automation Letters},
    doi = {10.1109/LRA.2026.3674671}
}
```

## Acknowledgements
This code largely benefits from the following repositories: [Transmorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), [Voxelmorph](https://github.com/voxelmorph/voxelmorph). Thanks to their authors for opening the source of their excellent works.
