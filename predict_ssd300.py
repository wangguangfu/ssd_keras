#!/Users/test/anaconda/bin/python
# -*- coding:utf-8 -*-
# @Time    : 2017/9/27
# @Author  : Guangfu Wang
# @Email   : guangfu.wang@ygomi.com
# @description:


from matplotlib import pyplot as plt
from keras_ssd300 import ssd_300
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator
import pylab
import os

img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 6 # Number of color channels of the input images
n_classes = 4 # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]] # The anchor box aspect ratios used in the original SSD300
two_boxes_for_ar1 = True
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True
batch_size = 8
classes = ['background', 'direction', 'string', 'divide']
model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                  n_classes=n_classes,
                                  min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                  max_scale=None,
                                  scales=scales,
                                  aspect_ratios_global=None,
                                  aspect_ratios_per_layer=aspect_ratios,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  coords=coords,
                                  normalize_coords=normalize_coords)

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.4,
                                neg_iou_threshold=0.1,
                                coords=coords,
                                normalize_coords=normalize_coords)

val_dataset = BatchGenerator(images_path='/Users/test/Documents/data/ssd_data/ipm-name-division/',
                             images_path_mul_channel='/Users/test/Documents/data/ssd_data/seg_result/forGuangfu/ipm-name1_colorlabel_crop/',
                             include_classes='all',
                             box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

val_dataset.parse_csv(labels_path='/Users/test/Documents/data/ssd_data/ch_label.csv',
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     random_crop=(300, 300, 1, 3),
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     diagnostics=False)

n_val_samples = val_dataset.get_n_samples()

predict_generator = val_dataset.generate(batch_size=1,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         random_crop=(300, 300, 1, 3),
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False)
model.load_weights('/Users/test/Documents/data/ssd_ckeckpoint/multi_channel_modified/ssd300_0_weights_epoch925_loss0.0023.h5',
                   by_name=False)

while True:
  X, y_true, filenames = next(predict_generator)
  i = 0  # Which batch item to look at

  print("Image:", filenames[i])
  print()
  print("Ground truth boxes:\n")
  print(y_true[i])


  y_pred = model.predict(X)
  print 'x',X.shape

  y_pred_decoded = decode_y(y_pred,
                            confidence_thresh=0.9,
                            iou_threshold=0.45,
                            top_k=20,
                            input_coords='centroids',
                            normalize_coords=normalize_coords,
                            img_height=img_height,
                            img_width=img_width)
  if False == y_pred_decoded:
    continue

  print("Predicted boxes:\n")
  print(y_pred_decoded[i])

  plt.figure(figsize=(20, 12))
  plt.imshow(X[i][:,:,0:3])
  current_axis = plt.gca()

  for box in y_pred_decoded[i]:
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(
      plt.Rectangle((box[2], box[4]), box[3] - box[2], box[5] - box[4], color='blue', fill=False, linewidth=2))
    current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor': 'blue', 'alpha': 1.0})

  for box in y_true[i]:
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(
      plt.Rectangle((box[1], box[3]), box[2] - box[1], box[4] - box[3], color='green', fill=False, linewidth=2))
    current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

  pylab.show()
  os.system('pause')


