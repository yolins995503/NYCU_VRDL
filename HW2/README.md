NYCU_VRDL_HW2  
  
Using the yolov5 to implement the object detection  
  
Step: Prepare Train Inference  
  
  Prepare :  
   
  1. Clone the source of yolov5  
     Command : git clone https://github.com/ultralytics/yolov5
    
  2. Install the environment  
     Command : pip install -r requirements.txt  
       
  3. Modify the （get_label.py） to declare the (training images dir)  
     (line 6 , training images dir , relative to 'path')  
     (line 6 , training images dir , relative to 'path')  
       
  4. Execute the (get_label.py) to obtain the training labels in txt (N txt for N images)  
     Command : python get_label.py  
       
     Input : digitStruct.mat  
     Output : training labels in labels folder  
       
  5. It is necessary to put (images) and (labels) in the same folder  
       
  6. Modify the (dataset.yaml) to declare the (training images dir) , (validation images dir) and (number of labels)  
     (line 2 , dataset root dir)  
     (line 3 , train images  , relative to 'path')   
     (line 4 , valid images  , relative to 'path')   
     (line 8 , number of classes)  
     (line 9 , class names)  
  
  Train :  
    
  1. Execute the (train.py) to training the model  
     Command : python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt  
     Select the images size (--img 640)  
                batch size (--batch 16)  
                epochs (--epochs 3)  
                training data (--data dataset.yaml)  
                pretrain model weight (--weights yolov5s.pt) There are four pretrain model weight (yolov5s.pt , yolov5m.pt , yolov5l.pt , yolov5x.pt) 
                  
     Output : model weight (best.pt , last.pt) and other training record in ( /run/train/exp ) folder  
  
    
  Inference : 
    
  1. Execute the (detect.py) to inference  
     Command : python detect.py --source /home/bsplab/Documents/yolin/VRDL_HW2/test/test  
     --weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt --conf 0.5  
     Select the test images dir (--source /home/bsplab/Documents/yolin/VRDL_HW2/test/test)  
                model weight (--weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.p)  
                threshold of confidence (--conf 0.5)  
                  
     Output : answer.json and visualization in ( /run/detect/exp ) folder  
     
  
Reference : https://github.com/ultralytics/yolov5  
