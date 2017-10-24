#!/Users/test/anaconda/bin/python
# -*- coding:utf-8 -*-
# @Time    : 2017/9/26
# @Author  : Guangfu Wang
# @Email   : guangfu.wang@ygomi.com
# @description: 将seg拼接图切割成300*300

import os
from PIL import Image

def ratioSize(img, model_input_size):
  (width, height) = img.size
  ratio = height/width
  return (model_input_size, int(ratio * model_input_size))

if __name__ == '__main__':
  data_dir = '/Users/test/Documents/data/ssd_data/seg_result/forSSD/merge/'
  out_dir = '/Users/test/Documents/data/ssd_data/seg_result/forSSD/crop/'
  or_size = 640
  crop_size = 300
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      print data_dir + file
      img_source = Image.open(data_dir + file)
      resize_img = img_source.resize(ratioSize(img_source, or_size))
      crop_img_big = resize_img.resize(ratioSize(resize_img, crop_size))
      # print img_source.size, resize_img.size, crop_img.size
      for i in range(crop_img_big.size[1]/ crop_size):
        box = (0, i * crop_size, crop_size, (i + 1) * crop_size)
        crop_img = crop_img_big.crop(box)
        crop_img_name = file[0:-4] + '_div_' + str(i) + '.png'
        # print crop_img_name
        crop_img.save(out_dir + crop_img_name)


