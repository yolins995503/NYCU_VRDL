from imantics import Mask, Image, Category, Dataset
import cv2
import os
import sys
import json
import os 
import argparse
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder create')
    else:
        print(path+'Already exist')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_folder" , type = str , default='/home/bsplab/Documents/yolin/VRDL_HW3/dataset/stage1_train')
    args = parser.parse_args()
    mkdir('coco')
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(1000)
    # exit()
    train_files = os.listdir(args.train_folder)
    image_list = []
    dataset = Dataset('Data Science Bowl 2018')
    for image_id, image_name in enumerate(train_files):
        # print(image_name)
        image = Image.from_path(args.train_folder+f'/{image_name}/images/{image_name}.png')
        image.id = image_id
        mask_files = os.listdir(args.train_folder+f'/{image_name}/masks')
        print(len(mask_files))
        for mask_name in mask_files:
            if mask_name.endswith('.png'):
                # print(mask_name)
                mask_array = cv2.imread(args.train_folder+f'/{image_name}/masks/{mask_name}', cv2.IMREAD_UNCHANGED)
                # print(mask_array.shape)
                mask = Mask(mask_array)
                categ = Category("nucleus")
                categ.id = 1
                image.add(mask, category=categ)
        image_coco_json = image.coco()
        with open(f'coco/{image_name}.json', 'w') as output_json_file:
            json.dump(image_coco_json, output_json_file, indent=4)
        dataset.add(image)
        # image.save(f'coco/{image_name}.json', style='coco')
    # dataset = Dataset('Data Science Bowl 2018', images=image_list)
    coco_json = dataset.coco()
    with open('coco/train.json', 'w') as output_json_file:
        json.dump(coco_json, output_json_file, indent=4)      
    print(len(train_files))
    
    with open('coco/train.json', newline='') as jsonfile:
        coco_json = json.load(jsonfile)
        annotations = coco_json["annotations"]
        # print(len(annotations))
        correct_seg = []
        for i in range(len(annotations)):
            if len(annotations[i]["segmentation"][0]) <= 4:
                print(annotations[i]["id"])
                continue
            correct_seg.append(annotations[i])
            coco_json["annotations"] = correct_seg
        with open('coco/train_correct.json', 'w') as output_json_file:
            json.dump(coco_json, output_json_file, indent=4)
            # print(len(annotations[0]["segmentation"][0]))