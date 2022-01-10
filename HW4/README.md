NYCU_VRDL_HW4
# NYCU_VRDL_HW4
### Inplement the Super Resolution
## Install Environment
Clone the source of IMDN
```
git clone https://github.com/Zheng222/IMDN
```
## Prepare the data
1. Split the data into train and valid  
```
```
2. Resize the orginal image into 1/3 size  
```
python resize.py
```

3. Convert to images to npy for training  
```
python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
```
4. Modify the path in DIV2k  
(line 44 , 45 ,46)
```
```
## Train
Run training x2 model
```
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 2 --pretrained checkpoints/IMDN_x2.pth
```
Run training x3 model
```
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 3 --pretrained checkpoints/IMDN_x3.pth
```
Run training x4 model
```
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 4 --pretrained checkpoints/IMDN_x4.pth
```

## Inference
Run testing x2 model
```
python test_IMDN.py --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2
```
Run testing x3 model
```
python test_IMDN.py --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 3
```
Run testing x4 model
```
python test_IMDN.py --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 4
```
## 5. Model weight link
https://drive.google.com/drive/folders/1QgEya5EyYnIUVBgNg5U7oog856U39_Bx?usp=sharing

## 6. Reference
[1] https://github.com/Zheng222/IMDN

