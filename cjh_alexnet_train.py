#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-26 下午8:03
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : cjh_alexnet_train.py
# @Software: PyCharm Community Edition
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
import numpy as np
import random as rd
import cv2,os,time,json,csv

model_save_path=r'/home/cjh/tensorflow-vgg16-train-and-test-master/model/alex_net'
csv_name='save_'+time.strftime('%Y_%m_%d_%H_%M',time.localtime())+'.csv'
csv_save_path=os.path.join(model_save_path,csv_name)
pic_size=227
num_step=100000
batch_size=5

def conv2d(x,w,b,strides=1,padding='SAME'):
    z1=tf.nn.conv2d(x,w,strides=[1, strides, strides, 1],padding=padding)
    z2=tf.nn.bias_add(z1,b)
    z3=tf.nn.relu(z2)
    return z3

def maxpool2d(x,ksize=2,stride=2,padding='SAME'):
    z1=tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)
    return z1

def full_connect(x,w,b,dropout):
    fc1=tf.add(tf.matmul(x,w),b)
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,dropout)
    return fc1

def init_var():
    weight_dict={
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([9216, 2048])),
        'wd2': tf.Variable(tf.random_normal([2048, 2048])),
        'out': tf.Variable(tf.random_normal([2048,2]))
    }
    bias_dict={
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([2048])),
        'bd2': tf.Variable(tf.random_normal([2048])),
        'out': tf.Variable(tf.random_normal([2]))
    }
    return weight_dict,bias_dict

def net_model(x,weight_dict,bias_dict,dropout):
    conv1=conv2d(x,weight_dict['wc1'],bias_dict['bc1'],strides=4,padding='VALID')
    pool1=maxpool2d(conv1,ksize=3,stride=2,padding='VALID')
    conv2=conv2d(pool1,weight_dict['wc2'],bias_dict['bc2'])
    pool2=maxpool2d(conv2,ksize=3,stride=2,padding='VALID')
    conv3=conv2d(pool2,weight_dict['wc3'],bias_dict['bc3'])
    conv4=conv2d(conv3,weight_dict['wc4'],bias_dict['bc4'])
    conv5=conv2d(conv4,weight_dict['wc5'],bias_dict['bc5'])
    pool3=maxpool2d(conv5,ksize=3,stride=2,padding='VALID')

    # fc0=tf.contrib.layers.flatten(pool3) #9216
    fc0 = tf.reshape(pool3, [-1, 6 * 6 * 256])

    fc1=full_connect(fc0,weight_dict['wd1'],bias_dict['bd1'],dropout)
    fc2=full_connect(fc1,weight_dict['wd2'],bias_dict['bd2'],dropout)
    out=tf.add(tf.matmul(fc2, weight_dict['out']), bias_dict['out'])
    # out layer no relu
    return out

def get_test_pic():
    path = r'/home/cjh/tensorflow-vgg16-train-and-test-master/cat_dog/test2'
    pic_list = []
    label_list = []
    for pic_name in os.listdir(path):
        pic_address = os.path.join(path, pic_name)
        img=cv2.imread(pic_address)
        img=cv2.resize(img,(pic_size,pic_size))/255
        pic_list.append(img)
        if pic_name.find('cat') != -1:
            label_list.append([1,0])
        elif pic_name.find('dog') != -1:
            label_list.append([0,1])
        else:
            raise BufferError
    return np.array(pic_list),np.array(label_list)

def get_pic_dogandcat():
    path = r'/home/cjh/tensorflow-vgg16-train-and-test-master/cat_dog/train'
    label_list=[]
    pic_list=[]
    for num in range(batch_size):
        label=rd.randint(0,1)
        if label==0:
            pic_name = 'cat.' + str(rd.randint(101, 12499)) + '.jpg'
            pic = cv2.imread(os.path.join(path, pic_name))
            pic = cv2.resize(pic, (pic_size, pic_size))
            pic_list.append(pic)
            label_list.append([1, 0])
        elif label==1:
            pic_name = 'dog.' + str(rd.randint(101, 12499)) + '.jpg'
            pic = cv2.imread(os.path.join(path, pic_name))
            pic = cv2.resize(pic, (pic_size, pic_size))
            pic_list.append(pic)
            label_list.append([0, 1])
    return np.array(pic_list),np.array(label_list)

def get_pic():
    path = r'/home/cjh/tensorflow-vgg16-train-and-test-master/picture'
    pic_list = []
    label_list = []
    for i in os.listdir(os.path.join(path, 'dog')):
        pic_address = os.path.join(path, 'dog', i)
        img=cv2.imread(pic_address)
        img=cv2.resize(img,(pic_size,pic_size))/255
        pic_list.append(img)
        label_list.append([1, 0])
    for i in os.listdir(os.path.join(path, 'cat')):
        pic_address = os.path.join(path, 'cat', i)
        img = cv2.imread(pic_address)
        img = cv2.resize(img, (pic_size, pic_size))/255
        pic_list.append(img)
        label_list.append([0, 1])
    return np.array(pic_list),np.array(label_list)

def change_pic(pic_list_org):
    pic_list = []
    change_color=30
    for pic_org in pic_list_org:
        rd_num=rd.randint(0,4)
        if rd_num == 0:
            # 保持不动
            pic_list.append(pic_org)
        elif rd_num == 1:
            # 水平翻转
            pic_list.append(cv2.flip(pic_org,1))
        elif rd_num == 2:
            # 增强红
            r, g, b = cv2.split(pic_org)
            r += rd.randint(1, change_color)
            g -= rd.randint(1, change_color)
            b -= rd.randint(1, change_color)
            pic_list.append(cv2.merge((r, g, b)))
        elif rd_num == 3:
            # 增强绿
            r, g, b = cv2.split(pic_org)
            r -= rd.randint(1, change_color)
            g += rd.randint(1, change_color)
            b -= rd.randint(1, change_color)
            pic_list.append(cv2.merge((r, g, b)))
        elif rd_num == 4:
            # 增强蓝
            r, g, b = cv2.split(pic_org)
            r -= rd.randint(1, change_color)
            g -= rd.randint(1, change_color)
            b += rd.randint(1, change_color)
            pic_list.append(cv2.merge((r, g, b)))
    return pic_list

def main():
    X = tf.placeholder(tf.float32, [None,pic_size,pic_size,3])
    Y = tf.placeholder(tf.float32, [None,2])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0,trainable=False)

    weight_dict,bias_dict=init_var()
#############################################################################
    conv1 = conv2d(X, weight_dict['wc1'], bias_dict['bc1'], strides=4, padding='VALID')
    pool1 = maxpool2d(conv1, ksize=3, stride=2, padding='VALID')
    conv2 = conv2d(pool1, weight_dict['wc2'], bias_dict['bc2'])
    pool2 = maxpool2d(conv2, ksize=3, stride=2, padding='VALID')
    conv3 = conv2d(pool2, weight_dict['wc3'], bias_dict['bc3'])
    conv4 = conv2d(conv3, weight_dict['wc4'], bias_dict['bc4'])
    conv5 = conv2d(conv4, weight_dict['wc5'], bias_dict['bc5'])
    pool3 = maxpool2d(conv5, ksize=3, stride=2, padding='VALID')

    # fc0=tf.contrib.layers.flatten(pool3) #9216
    fc0 = tf.reshape(pool3, [-1, 6 * 6 * 256])

    fc1 = full_connect(fc0, weight_dict['wd1'], bias_dict['bd1'], keep_prob)
    fc2 = full_connect(fc1, weight_dict['wd2'], bias_dict['bd2'], keep_prob)
    out = tf.add(tf.matmul(fc2, weight_dict['out']), bias_dict['out'])
    # out layer no relu

#############################################################################

    # logits=net_model(X,weight_dict=weight_dict,bias_dict=bias_dict,dropout=keep_prob)
    logits=out
    prediction=tf.nn.softmax(logits=logits)

    loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))

    learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.9, staircase=True)
    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op,global_step=global_step)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init=tf.global_variables_initializer()
    test_pic_list,test_label_list=get_test_pic()
    with tf.Session() as sess:
        sess.run(init)
        pic_list, label_list = get_pic_dogandcat()

        for iteration in range(1,num_step+1):
            start = time.time()
            end = time.time()

            # print('read_time=',end-start,end='')
            # train_pic_list=change_pic(pic_list_org)
            start=time.time()
            sess.run(train_op,feed_dict={X:pic_list,Y:label_list,keep_prob:0.8})
            end=time.time()
            # print('train_time=',end-start)
            # print(sess.run(global_step),sess.run(learning_rate))
            if iteration % 5 ==0 or iteration==1:
                lr,loss, acc = sess.run([learning_rate,loss_op, accuracy], feed_dict={X: pic_list,
                                                                     Y: label_list,
                                                                     keep_prob: 1.0})
                logits_output=sess.run(logits,feed_dict={X:pic_list,Y:label_list,keep_prob:1.0})
                prediction_output=sess.run(prediction,feed_dict={X:pic_list,Y:label_list,keep_prob:1.0})
                x_output=sess.run(X,feed_dict={X:pic_list,Y:label_list,keep_prob:1.0})
                conv1_output=sess.run(conv1,feed_dict={X:pic_list,Y:label_list,keep_prob:1.0})
                print("Step " + str(iteration) +
                      ", Minibatch Loss= " +"{:.8f}".format(loss) +
                      ", Training Accuracy= " +"{:.5f}".format(acc) +
                      ', learning_rate= '+"{:.8f}".format(lr)
                      )
                print(logits_output)
                print(prediction_output)
                print(x_output)
                print(conv1_output)
                with open(csv_save_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, loss, acc])

                if iteration % 25 == 0:
                    pic_list, label_list = get_pic_dogandcat()
                if iteration % 10000 ==0:
                    t=time.strftime('%Y_%m_%d_%H_%M',time.localtime())
                    model_name=t+'model.ckpt'
                    model_path=os.path.join(model_save_path,model_name)
                    saver = tf.train.Saver()
                    saver.save(sess, save_path=model_path)
                if iteration % 1000 ==0:
                    print('123')

        print("Optimization Finished!")
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_pic_list,Y: test_label_list,keep_prob: 1.0})
        print("Finally " + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        model_name = t + 'model.ckpt'
        model_path = os.path.join(model_save_path, model_name)
        saver = tf.train.Saver()
        saver.save(sess, save_path=model_path)
        print("Model Save Finished!")

if __name__ == '__main__':
    main()
