import cv2
import os
import json

def make_output_folder(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    ALB_path = output_path+'/ALB'
    if not os.path.isdir(ALB_path):
        os.makedirs(ALB_path)
    BET_path = output_path+'/BET'
    if not os.path.isdir(BET_path):
        os.makedirs(BET_path)
    DOL_path = output_path+'/DOL'
    if not os.path.isdir(DOL_path):
        os.makedirs(DOL_path)
    LAG_path = output_path+'/LAG'
    if not os.path.isdir(LAG_path):
        os.makedirs(LAG_path)
    OTHER_path = output_path+'/OTHER'
    if not os.path.isdir(OTHER_path):
        os.makedirs(OTHER_path)
    SHARK_path = output_path+'/SHARK'
    if not os.path.isdir(SHARK_path):
        os.makedirs(SHARK_path)
    YFT_path = output_path+'/YFT'
    if not os.path.isdir(YFT_path):
        os.makedirs(YFT_path)

def crop_image(train_img_dir ,json_name , output_path ):
    with open(json_name) as f:
        data = json.load(f)
        print (len(data) )
        for index in range(len(data)):
            img_name = data[index]['filename']
            img_dir = train_img_dir + '/' + img_name
            # print (img_dir)
            for idx in range(len(data[index]['annotations'])):
                x = int(data[index]['annotations'][idx]['x'])
                y = int(data[index]['annotations'][idx]['y'])
                width = int(data[index]['annotations'][idx]['width'])
                height = int(data[index]['annotations'][idx]['height'])
                # print(img_dir)
                img = cv2.imread(img_dir)
                crop_image = img[y:y+height , x:x+width]
                cv2.imwrite(output_path+'/'+img_name.split('.')[0]+'_'+str(idx)+'.jpg' , crop_image)



if __name__=='__main__':
    output_path = ('train_crop')
    make_output_folder(output_path)
    train_img_dir = ('train')

    crop_image(train_img_dir , 'label_json/ALB.json' , output_path)
    crop_image(train_img_dir , 'label_json/BET.json' , output_path)
    crop_image(train_img_dir , 'label_json/DOL.json' , output_path)
    crop_image(train_img_dir , 'label_json/LAG.json' , output_path)
    crop_image(train_img_dir , 'label_json/OTHER.json' , output_path)
    crop_image(train_img_dir , 'label_json/SHARK.json' , output_path)
    crop_image(train_img_dir , 'label_json/YFT.json' , output_path)




    