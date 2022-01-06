NYCU_VRDL_HW4
# NYCU_VRDL_HW4
## 1.Install Environment
Clone the source of VDSR
```
git clone https://github.com/2KangHo/vdsr_pytorch.git
```
## 2.Split the dataset into train and valid
    <NAME>/train/ ,  <NAME>/valid/
    
## 3. Train the model
Execute the main.py or run the main.sh
    
```
python main.py --dataset DF2K --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --batch_size 128 --test_batch_size 32 --epochs 100
```
    
```
bash main.sh
```

## 4. Inference
Execute the run.py for single images
```
> python run.py --cuda --gpuids 0 1 --scale_factor 2 --model model_epoch_100.pth --input_image test_scale2x.jpg --output_filename test_scale2x_out.jpg
```
Run the all images
```
bash run.sh
```

## 5. Model Link
https://drive.google.com/drive/folders/1QgEya5EyYnIUVBgNg5U7oog856U39_Bx?usp=sharing

## 6. Reference
[1] https://github.com/2KangHo/vdsr_pytorch

