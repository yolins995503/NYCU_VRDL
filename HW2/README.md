NYCU_VRDL_HW2
# NYCU_VRDL_HW2
Using the yolov5 to implement the object detection

## Install Environment
1. Clone the source of yolov5
```
git clone https://github.com/ultralytics/yolov5
```
2. Install the environment
```
pip install -r requirements.txt
```
3. Modify the （get_label.py）to declare the (training images dir)

    (line 6 , training images dir , relative to 'path')
    
    (line 6 , training images dir , relative to 'path')
```
```
4. Execute the (get_label.py) to obtain the training labels in txt (N txt for N images)
```
 python get_label.py
```
5. It is necessary to put (images) and (labels) in the same folder
```
```
6. Modify the (dataset.yaml) to declare the (training images dir) , (validation images dir) and (number of labels)

    (line 2 , dataset root dir)
    
    (line 3 , train images , relative to 'path')
    
    (line 4 , valid images , relative to 'path')
    
    (line 8 , number of classes)
    
    (line 9 , class names)
```
```
## Train
Select the images size (--img 640)
batch size (--batch 16)
epochs (--epochs 3)
training data (--data dataset.yaml)
pretrain model weight (--weights yolov5s.pt) 
There are four pretrain model weight (yolov5s.pt , yolov5m.pt , yolov5l.pt , yolov5x.pt)
```
 python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt
```

## Inference
1. Execute the (detect.py) to inference

    Select the test images dir (--source /home/bsplab/Documents/yolin/VRDL_HW2/test/test)
    
    model weight (--weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt)
    
    threshold of confidence (--conf 0.5)

```
python detect.py --source /home/bsplab/Documents/yolin/VRDL_HW2/test/test
--weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt --conf 0.5
```

2. Execute the (inference.ipynb) to obtain the execution time per images

    It is necessary to upload the (reauirement.txt) , model_weight(best.pt) , testing image
    
    Execute the Step2 to install the environment (!pip install -r /content/yolov5/requirements.txt)


    Execute the last cell of step4 (!python detect.py --source /content/test_images/test --weights /content/best.pt --conf 0.1)
    
```
```
 
## Model weight link
 https://drive.google.com/file/d/1dFj4zf9DcsjIWOl2Q09XG4vuzzd2TDXo/view?usp=sharing
 
## Reference
 https://github.com/ultralytics/yolov5