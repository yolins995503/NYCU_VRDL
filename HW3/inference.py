import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import pycocotools._mask as _mask
import os
import mmcv

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train6",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.TEST.DETECTIONS_PER_IMAGE = 2000
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
iou         = _mask.iou
merge       = _mask.merge
frPyObjects = _mask.frPyObjects

def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]

def area(rleObjs):
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]

def toBbox(rleObjs):
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]

def write_answer(annotations, im , img_id):
    outputs = predictor(im)
    # print(len(outputs['instances']))
    for i in range(len(outputs['instances'])):
        bbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()[i]
        # print(outputs["instances"].pred_boxes.tensor.cpu().numpy()[i])
        x = (bbox[0])
        y = (bbox[1])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # print(x)
        # print(y)
        # print(w)
        # print(h)
        # print(outputs["instances"].scores[i].cpu().numpy())
        # print(outputs["instances"].pred_masks[0])
        rls = encode(np.asfortranarray(outputs["instances"].pred_masks.cpu().numpy()[i]))
        # print(rls['size'])
        # print(rls['counts'])
        # print( outputs["instances"].pred_masks.cpu().numpy()[i] )
        # cv2_imshow(outputs["instances"].pred_masks.cpu().numpy()[i]*255)
        seg = dict (
            size = rls['size'],
            counts = str(rls['counts'])
        )
        data_anno = dict (
            image_id = int(img_id),
            bbox = [int(x), int(y), int(w) , int(h)],
            score = outputs["instances"].scores[i].cpu().numpy(),
            category_id = int (1),
            segmentaion = seg
        )
        annotations.append(data_anno)
    return  annotations


annotations = []

im = cv2.imread('/content/ballon/test/TCGA-A7-A13E-01Z-00-DX1.png')
annotations = write_answer(annotations , im , img_id = 1)

im = cv2.imread('/content/ballon/test/TCGA-50-5931-01Z-00-DX1.png')
annotations = write_answer(annotations , im , img_id = 2)

im = cv2.imread('/content/ballon/test/TCGA-G2-A2EK-01A-02-TSB.png')
annotations = write_answer(annotations , im , img_id = 3)

im = cv2.imread('/content/ballon/test/TCGA-AY-A8YK-01A-01-TS1.png')
annotations = write_answer(annotations , im , img_id = 4)

im = cv2.imread('/content/ballon/test/TCGA-G9-6336-01Z-00-DX1.png')
annotations = write_answer(annotations , im , img_id = 5)

im = cv2.imread('/content/ballon/test/TCGA-G9-6348-01Z-00-DX1.png')
annotations = write_answer(annotations , im , img_id = 6)

mmcv.dump(annotations, 'answer.json', indent=4)