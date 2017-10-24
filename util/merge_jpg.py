#!/Users/test/anaconda/bin/python
# -*- coding:utf-8 -*-
# @Time    : 2017/9/29
# @Author  : Guangfu Wang
# @Email   : guangfu.wang@ygomi.com
# @description:

import os
from PIL import Image

def ratio_size(img, model_input_size):
  (width, height) = img.size
  ratio = height/width
  return int(ratio * model_input_size)

def merge_img(img_width, img_height, counter, img_file):
  target = Image.new('RGB', (img_width, img_height * counter))
  left = 0
  right = img_height
  for image in img_file:
    target.paste(image, (0, left, img_width, right))
    left += img_height  # 从上往下拼接，左上角的纵坐标递增
    right += img_height  # 左下角的纵坐标也递增　
  return target

if __name__ == '__main__':
  data_dir = '/Users/test/Documents/data/labelled_painting/paint_image_new/'
  out_dir = '/Users/test/Documents/data/labelled_painting/paint_image_new_merge/'
  img_file = []
  counter = 0
  img_width = 640
  img_height = 1280
  for root, dirs, files in os.walk(data_dir):
    for i in range(len(files)):
      print files[i]
      if files[i].split('.')[0][-1] == '1' and len(img_file) != 0:
        target = merge_img(img_width, img_height, counter, img_file)
        quality_value = 100
        target.save(out_dir + files[i-1].split('.')[0][0:-2] + '.jpg', quality=quality_value)
        del img_file[:]
        counter = 0
      img_file.append(Image.open(data_dir + files[i]))
      counter += 1
      if i+1 == len(files):
        target.save(out_dir + files[i].split('.')[0][0:-2] + '.jpg', quality=quality_value)



