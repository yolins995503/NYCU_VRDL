# NYCU VRDL Final  

## 1. Introduction
Topic : The Nature Conservancy Fisheries Monitoring  

Link : https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

We Use classification model and objection detection model to distinguish the fishes.  

## 2. Training code
Procedure:

1.  Crop the training data into many training images by using the (label.json)  
2.  Train the classification model by training images  
3.  Train the objection detection model by training images  
4.  Detect the boundary box of fishes in testing data and crop into testing images  
5.  Predict the class of testing images  
6.  Ensemble models and clip the final results

Crop the training data into many training images by using the (label.json):  
```
python crop_dataset.py 
```
   
Train the classification model by training images:

```
python train_crop.py --model=resnet50
python train_crop.py --model=regnet_x_8gf
```

Train the objection detection model by training images:  
     
1. Clone the source of yolov5  
    ```
    git clone https://github.com/ultralytics/yolov5  
    ```

2. Install the environment  

    ```
    pip install -r requirements.txt  
    ```

3. Modify the（get_label.py）to declare the (training images dir)  
   (line 6 , training images dir , relative to 'path')  
   (line 6 , training images dir , relative to 'path')  

4. Execute the (label_mask.py) to obtain the training labels in txt (N txt for N images)

    ```
    python label_mask.py  
    ```

5. It is necessary to put (images) and (labels) in the same folder  

6. Modify the (dataset.yaml) to declare the (training images dir) , (validation images dir) and (number of labels)
   (line 2 , dataset root dir)  
   (line 3 , train images , relative to 'path')  
   (line 4 , valid images , relative to 'path')  
   (line 8 , number of classes)  
   (line 9 , class names)  

 7. Execute the (train.py) to training the model
    ```
    python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt
    ``` 
    Output : model weight (best.pt , last.pt) and other training record in ( /run/train/exp ) folder  
            
## 3. Inference code
You can download [model weight](https://drive.google.com/drive/folders/104ZJATHoQJcIoAiDLS3PKJUOY3oAirGN?usp=sharing) included yolov5, ResNet50 and RegNet. And put the model weights in `model` directory.

Detect the boundary box of fishes in testing data and crop into testing images:

```
python detect.py --weights=model/best.pt --source=test_stg1/ --save-crop
python detect.py --weights=model/best.pt --source=test_stg2/ --save-crop
```
          
Predict the class of testing images:

```
python inference_crop.py --model=regnet_x_8gf --output=regnet_x_8gf_crop.csv
python inference_crop.py --model=resnet50 --output=resnet50_epoch10_crop.csv
```

Ensemble models and clip the final results:

```
python ensemble.py
```
    
## 4. Pre-trained models
Please download [model weight](https://drive.google.com/drive/folders/104ZJATHoQJcIoAiDLS3PKJUOY3oAirGN?usp=sharing) included yolov5, ResNet50 and RegNet. And put the model weights in `model` directory.

## 5. Reference
[1] yolov5: https://github.com/ultralytics/yolov5  
[2] object detection label dataset1: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25428  
[3] object detection label dataset2: https://github.com/autoliuweijie/Kaggle/tree/master/NCFM/datasets  
[4] K-fold Cross-Validation: https://github.com/lidxhaha/Kaggle_NCFM  
[5] InceptionV3 network : https://github.com/pengpaiSH/Kaggle_NCFM
