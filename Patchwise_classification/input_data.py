import tensorflow as tf
import numpy as np
import os
import random
from skimage import io
import pandas as pd

# all pixels are subtracted by the mean RGB value of all WSI
# MEAN_RGB = np.asarray([216.82551656994778, 115.08248482729931, 209.54666851831385])
MEAN_RGB = np.asarray([217, 115, 210], dtype='uint8')


# returns a dictionary to map the image name to the classes they belong to.
def get_feature_dict(path, feature_name):
    '''
    :param path: path of the csv file which contains the data description
    :param feature_name: the feature to classify
    :return: a dictionary to map the image name to the classes they belong to.
    '''
    df = pd.read_csv(path)
    table = {}
    for i in range(len(df)):
        table[df.loc[i, 'ImageIndex']] = df.loc[i, feature_name]
    return table


# a method used to read patches along with the according label in batches
def read_local_data(data_dir, batch_size, step, feature_dict,
                    name_list, n_classes, img_shape=[224, 224, 3]):
    '''
    :param data_dir: a directory where the data is stored
    :param batch_size:
    :param step: the current step number
    :param feature_dict: a dictionary to map the image name to the classes they belong to.
    :param name_list: a list of paths of all patches
    :param n_classes: number of classes to classify
    :param img_shape:
    :return:
    '''
    while step * batch_size + batch_size > len(name_list):
        step -= 1
    step = int(step)
    batch_list = name_list[step * batch_size:step * batch_size + batch_size]
    images = np.zeros(shape=[batch_size, img_shape[0], img_shape[1], img_shape[2]])
    labels = np.zeros(shape=[batch_size], dtype=int)
    index = 0

    for file_name in batch_list:
        path = file_name
        image = np.asarray(io.imread(path))
        image -= MEAN_RGB
        images[index] = image
        base_name = os.path.basename(file_name)
        image_index = int(base_name[12:16])
        labels[index] = feature_dict[image_index]
        index += 1

    labels = tf.one_hot(labels, depth=n_classes, dtype=tf.int16)
    return images, labels


# a method to give a list of all patches that satisfy the requirement
def get_full_list(cluster, folder, folder_clustermaps, csv='D:\\data\\1-10\\data.csv', data_type=None, data_name=None):
    '''
    :param cluster: the number of cluster
    :param folder: a directory where all patches is stored
    :param folder_clustermaps: a directory where all cluster maps are stored
    :param csv: the path of data description file
    :param data_type: the type of data asked for ('train' or 'val' or 'test')
    :param data_name: the image name asked for
    :return:
    '''
    if data_type is not None:
        list_image = os.listdir(folder)
        df = pd.read_csv(csv)
        df = df.set_index('ImageIndex')
        full_list = []
        for image in list_image:
            image_index = int(image[12:16])
            path_image = os.path.join(folder, image)
            if df.loc[image_index]['data_type'] == data_type:
                cluster_map = os.path.join(folder_clustermaps, image[:16] + '.npy')
                cluster_map = np.load(cluster_map)
                loc =[]
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
                    full_list.append(path_tile)
        np.random.shuffle(full_list)
    elif data_name is not None:
        df = pd.read_csv(csv)
        df = df.set_index('ImageIndex')
        full_list = []
        list_image = data_name
        for image in list_image:
            image_index = int(image[12:16])
            path_image = os.path.join(folder, image)

            cluster_map = os.path.join(folder_clustermaps, image[:16] + '.npy')
            cluster_map = np.load(cluster_map)
            loc = []
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
                full_list.append(path_tile)
        np.random.shuffle(full_list)
    return full_list

def get_full_list_submission(folder):
    list_image = os.listdir(folder)
    full_list = []
    for image in list_image:
        image_index = int(image[12:16])
        path_image = os.path.join(folder, image)
        list_tiles = os.listdir(path_image)
        for tile in list_tiles:
            path_tile = os.path.join(path_image, tile)
            full_list.append(path_tile)
    return full_list


