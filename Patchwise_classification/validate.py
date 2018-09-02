
import os
import os.path
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import input_data
import VGG
import tools
from random import shuffle

# configurations of CNN
IMG_W = 224
IMG_H = 224
IMG_D = 3
N_CLASSES = 2
BATCH_SIZE = 48
learning_rate = 0.001
MAX_EPOCH = 3  # it took me about one hour to complete the training.
TRAINABLE = True  # True for training from scratch and deep tuning, False for shallow tuning.
IS_FINETUNING = False

feature_to_classify = '5yrs'
data_dir = './Testing-dataset/test_data'  # the path where all patches are stored

# choose clustering model and number of cluster
ckpt_path='./Pretrained-model/CNN_model/model_0_10000.ckpt-10000'
folder_clustermaps = './Pretrained-model/Cluster_maps/cluster_maps_5'
cluster = 2


# perform entire-validation-set validation of a trained model
def tile_validation(ckpt_path, cluster, folder_clustermaps):
    '''
    :param ckpt_path: the trained model to evaluate
    :param cluster: the cluster to validate
    :param folder_clustermaps:
    :return:
    '''
    feature_dict = input_data.get_feature_dict('./Testing-dataset/data.csv', '5yrs')

    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))

    logits = VGG.VGG16_nobn(x, N_CLASSES, keep_prob=1, is_pretrain=False)

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('restoring weight')
        saver.restore(sess, ckpt_path)
        # list_val = input_data.get_full_list(data_type='val', cluster=cluster,
        #                                     folder=data_dir, folder_clustermaps=folder_clustermaps)
        list_val = input_data.get_full_list_submission(folder=data_dir)
        # list_val = input_data.get_full_list(data_name=['2018_06_01__5836'] , cluster=cluster,
        #                                     folder=data_dir, folder_clustermaps=folder_clustermaps)
        max_step = int(len(list_val) / BATCH_SIZE)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for step in np.arange(max_step):
            if step % 10 == 0:
                print('%d of %d' % (step, max_step))

            val_image_batch, val_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                          batch_size=BATCH_SIZE,
                                                                          step=step,
                                                                          feature_dict=feature_dict,
                                                                          n_classes=N_CLASSES,
                                                                          name_list=list_val)
            val_labels = sess.run(val_label_batch)
            val_images = val_image_batch
            val_logits = sess.run(logits, feed_dict={x: val_images})

            for label, logit in zip(val_labels, val_logits):
                if label[0] == 1 and logit[0] >= 0.5:
                    tn += 1
                elif label[1] == 1 and logit[1] >= 0.5:
                    tp += 1
                elif label[0] == 1 and logit[1] >= 0.5:
                    fp += 1
                else:
                    fn += 1

        print('tn=%d    tp=%d   fp=%d   fn=%d' % (tn, tp, fp, fn))

        acc = (tn + tp)/(tn + tp + fp + fn)*100
        print('accuracy = %.2f' % (acc))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print('precision: %.2f  recall: %.2f   f1: %.2f' % (precision, recall, f1))


if __name__ == '__main__':
    tile_validation(ckpt_path=ckpt_path,
                    cluster=cluster,
                    folder_clustermaps=folder_clustermaps)