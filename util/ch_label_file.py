
#!/Users/test/anaconda/bin/python
# -*- coding:utf-8 -*-
# @Time    : 2017/9/28
# @Author  : Guangfu Wang
# @Email   : guangfu.wang@ygomi.com
# @description:

if __name__ == '__main__':
  csv_file = open('/Users/test/Documents/data/ssd_data/new_label.csv')
  out_file = open('/Users/test/Documents/data/ssd_data/ch_label.csv','w')
  for line in csv_file:
    print line[-3],line[0:-2]
    if line[-3] == '0':
      out_file.write(line[0:-4]+',3'+'\n')
      continue
    out_file.write(line[0:-2]+'\n')
