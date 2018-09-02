
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
import progressbar
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
data_dir = 'D:\\data\\1-10\\data_cluster1'  # the path where all patches are stored
pre_trained_weights = './vgg16_weights/vgg16.npy'  # the path of pretrained model

# choose clustering model and number of cluster
ckpt_path='C:\\Users\\xy31\\PycharmProjects\\VGG\\logs_nobn_IDC3_2\\train\\model_0_4000.ckpt-4000'
folder_clustermaps = 'D:\\data\\1-10\\cluster_maps_id_3'
cluster = 2
output_dict = 'D:\\data\\1-10\\predictions\\logs_nobn_PhC3_2\\'


# save the patch-wise predictions for WSIs.
def get_patch_prediction(ckpt_path, cluster, folder_clustermaps, output_dict):
    '''
    :param ckpt_path: the path of trained model
    :param cluster: the cluster to validate
    :param folder_clustermaps: a directory where the data is stored
    :param output_dict: output directory to save the predictions
    :return:
    '''
    feature_dict = input_data.get_feature_dict('D://data//1-10//data.csv', '5yrs')
    list_image = os.listdir(data_dir)
    # shuffle(list_image)

    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)

    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))

    logits = VGG.VGG16_nobn(x, N_CLASSES, keep_prob=1, is_pretrain=False)

    saver = tf.train.Saver(tf.global_variables())

    predictions = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('restoring weights')
        saver.restore(sess, ckpt_path)

        count0 = 0
        count1 = 1

        prog = 0
        for image in list_image:
            prog += 1
            tile_path_list = []
            image_index = int(image[12:16])
            image_name = image[12:]
            path_image = os.path.join(data_dir, image)
            cluster_map_path = os.path.join(folder_clustermaps, image[:16] + '.npy')
            cluster_map = np.load(cluster_map_path)
            loc = []
            predictions = []
            for h in range(len(cluster_map)):
                for w in range(len(cluster_map[h])):
                    if cluster_map[h][w] == cluster:
                        w_str, h_str = str(w), str(h)
                        loc.append([h_str, w_str])
            for pair in loc:
                while len(pair[0]) < 3:
                    pair[0] = '0' + pair[0]
                while len(pair[1]) < 3:
                    pair[1] = '0' + pair[1]
                tile_name = image[:16] + image[16:] + '_' + pair[0] + '_' + pair[1] + '.png'
                path_tile = os.path.join(path_image, tile_name)
                tile_path_list.append(path_tile)

            max_step = int(len(tile_path_list) / BATCH_SIZE)

            for step in range(max_step):
                print('%d of %d: finished %.2f%% ' % (prog, len(list_image), step / max_step * 100), end='\r', flush=True)
                val_image_batch, val_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                              batch_size=BATCH_SIZE,
                                                                              step=step,
                                                                              feature_dict=feature_dict,
                                                                              n_classes=N_CLASSES,
                                                                              name_list=tile_path_list)
                val_labels = sess.run(val_label_batch)
                val_images = val_image_batch
                val_logits = sess.run(logits, feed_dict={x: val_images})

                # logits = np.reshape(val_logits[1], (1, -1))

                for i in val_logits:
                    if i[0] >= 0.5:
                        count0 += 1
                    else:
                        count1 += 1

                predictions.extend(val_logits[:, 1])
            predictions = np.asarray(predictions)
            print(image_index, np.mean(predictions))
            print(count0, count1)
            output_path = os.path.join(output_dict, image_name)
            np.save(output_path, np.asarray(predictions))


if __name__ == '__main__':
    get_patch_prediction(ckpt_path=ckpt_path,
                         cluster=cluster,
                         folder_clustermaps=folder_clustermaps,
                         output_dict=output_dict)