# Some basic setup:
# Setup detectron2 logger
# python inference.py --yaml=mask_rcnn_X_101_32x8d_FPN_3x.yaml --model=mask_rcnn_X_101/model_final.pth
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import pycocotools.mask as mask_util
import argparse

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": int(1),
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results

def write_answer(im , img_id):
    outputs = predictor(im)
    return instances_to_coco_json(outputs["instances"].to("cpu"), img_id)

if __name__=='__main__':
    register_coco_instances("my_dataset_train1", {}, "train_correct.json", "train")
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("my_dataset_train1",)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.OUTPUT_DIR =("./output")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_3300.pth")  # path to the model we just trained
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    annotations = []

    im = cv2.imread('stage1_test/TCGA-A7-A13E-01Z-00-DX1.png')
    annotations.extend(write_answer(im , img_id = 1))

    im = cv2.imread('stage1_test/TCGA-50-5931-01Z-00-DX1.png')
    annotations.extend(write_answer(im , img_id = 2))

    im = cv2.imread('stage1_test/TCGA-G2-A2EK-01A-02-TSB.png')
    annotations.extend(write_answer(im , img_id = 3))

    im = cv2.imread('stage1_test/TCGA-AY-A8YK-01A-01-TS1.png')
    annotations.extend(write_answer(im , img_id = 4))

    im = cv2.imread('stage1_test/TCGA-G9-6336-01Z-00-DX1.png')
    annotations.extend(write_answer(im , img_id = 5))

    im = cv2.imread('stage1_test/TCGA-G9-6348-01Z-00-DX1.png')
    annotations.extend(write_answer(im , img_id = 6))

    answer_json = json.dumps(annotations, indent=4)
    with open("answer.json", "w") as outfile:
        outfile.write(answer_json)