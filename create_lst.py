#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-25 下午4:13
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : create_lst.py
# @Software: PyCharm
import os,random
def create_lst(path):
    pic_list=[x for x in os.listdir(path) if x.find('.jpg')!=-1]
    random.shuffle(pic_list)
    with open('test.lst','w+') as f:
        num_pic=0
        for pic_name in pic_list:
            if pic_name.find('cat')!=-1:
                label=0
            else:
                label=1
            need_write=str(num_pic)+'\t'+str(label)+'\t'+os.path.join(path,pic_name)+'\n'
            f.write(need_write)
            num_pic+=1
if __name__ == '__main__':
    create_lst('/media/cjh/colorful/Data_set/dog_and_cat/test2')