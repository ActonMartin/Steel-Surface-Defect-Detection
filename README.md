# Steel-Surface-Defect-Detection
This project involves performing semantic segmentation on pictures of steel surfaces to classify and localize the surface defects using the UNET architecture. The dataset for this project is taken from [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection) which contains about 12000 train images and 6000 test images(more information is available on the Kaggle website).

## Files Present

```
rootdir
    ├── UNET_128x800_image/
    │    ├── dataloader.py
    │    ├── model.py
    │    └── train.py
    │
    ├── UNET_64x400_image/
    │    ├── dataloader.py
    │    ├── model.py
    │    └── train.py
    │
    └── README.md
```

## About the Code:
The images in the dataset are of size 256x1600. The folder **UNET_128X800_image** contains the code which resizes the image to 128x800 and performs the segmentation task while the folder **UNET_64X400_image** contains the code which resizes the image to 64x400 and does the same as before. There are three codes dataloader.py, model.py and train.py in which train.py should be run to start the training.


Wandb is used here to log the losses, accurcies, scores and some input images along with its outputs. The wandb report can be referred [here](https://wandb.ai/manoj-s/Steel_Defect_Detection?workspace=user-manoj-s).

## Results Tabulation
![Alt Text](https://drive.google.com/file/d/1ya_B-9l4yYNetc_GEZYdI0d5c7ujADVE/view?usp=sharing)
