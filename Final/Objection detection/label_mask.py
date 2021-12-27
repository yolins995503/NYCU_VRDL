import h5py
import glob
import cv2 
import json 

def cls_name(name_string):
    if name_string == 'ALB':
        return 0
    if name_string == 'BET':
        return 1
    if name_string == 'DOL':
        return 2
    if name_string == 'LAG':
        return 3
    if name_string == 'OTHER':
        return 4
    if name_string == 'SHARK':
        return 5
    if name_string == 'YFT':
        return 6

def json_to_yolo(train_img_dir,json_name):
    with open(json_name) as f:
        data = json.load(f)
        print (len(data) )
        # print ( len(data[0]['annotations']) )
        for index in range(len(data)):
            file_name = (data[index]['filename'])
            txt_file_name = file_name.split('/')[1].split('.')[0]
            txt_file_name_path = "training_dataset_for_yolo/labels/"+ txt_file_name+'.txt'
            with open(txt_file_name_path, 'w') as f1:
                for idx in range(len(data[index]['annotations'])):
                    # print (train_img_dir + '/' + file_name)
                    img =  cv2.imread(train_img_dir + '/' + file_name)
                    # cv2.imwrite("iii.png" , img)
                    sp = img.shape
                    img_width = sp[1]
                    img_height = sp[0]
                    # print (img_width ,img_height)
                    cls = cls_name( (data[index]['annotations'][idx]['class']) )
                    x_center =( ( (data[index]['annotations'][idx]['x']) + ( (data[index]['annotations'][idx]['width']) ) /2 ) )/ img_width
                    y_center =( ( (data[index]['annotations'][idx]['y']) + ( (data[index]['annotations'][idx]['height']) ) /2) )/ img_height
                    width = (data[index]['annotations'][idx]['width']) / img_width
                    height = (data[index]['annotations'][idx]['height']) / img_height
                    f1.writelines(str(cls)+" "+str(x_center)+ " " +str(y_center)+" "+str(width)+" "+str(height)+'\n')
                    # if cls == 0:
                    #     f1.writelines(str(0)+" "+str(x_center)+ " " +str(y_center)+" "+str(width)+" "+str(height)+'\n')
                    # else:
                    #     f1.writelines(str(1)+" "+str(x_center)+ " " +str(y_center)+" "+str(width)+" "+str(height)+'\n')


if __name__=='__main__':
    train_img_list = glob.glob('/home/bsplab/Documents/yolin/VRDL_Final/preprocessing/label_json/*.json')
    train_img_dir = ('/home/bsplab/Documents/yolin/VRDL_Final/the-nature-conservancy-fisheries-monitoring/train/train')
 
    json_to_yolo('label_json/ALB.json')
    json_to_yolo('label_json/BET.json')
    json_to_yolo('label_json/DOL.json')
    json_to_yolo('label_json/LAG.json')
    json_to_yolo('label_json/OTHER.json')
    json_to_yolo('label_json/SHARK.json')
    json_to_yolo('label_json/YFT.json')


