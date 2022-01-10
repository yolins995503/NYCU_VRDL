NYCU_VRDL_HW1
# NYCU_VRDL_HW1
## Image processing
image_processing.py
Include the basic image processing functions

## Train
Contain the dataloader , model architecture , evaluation and training.

Input :  training images file & label , validation images file & label , model weight
Output : CNN_model.pt
```
python main.py
```

## Inference
Reproduce the submission file (answer.txt)

Input : CNN_model.pt , testing_img_order.txt , classes.txt
Output : result.txt , answer.txt

```
python inference.py
```

## Model weight link
 https://drive.google.com/file/d/19b_a6l8XbEFKeg-gTOYtItrLse2SzcMq/view?usp=sharing
