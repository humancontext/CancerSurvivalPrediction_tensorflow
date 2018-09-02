# TO Train and test:
# 0. get data ready, get paths ready !!!
# 1. run training_and_val.py and call train() in the console
# 2. call evaluate() in the console to test

# tensorboard --logdir=.//logs//
# http://localhost:6006
# 2018_06_01__5792_20_51_0.png
# 012345678901234567890123456789
# %%

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

# %%
IMG_W = 224
IMG_H = 224
IMG_D = 3
N_CLASSES = 2
BATCH_SIZE = 48
learning_rate = 0.001
MAX_EPOCH = 3  # it took me about one hour to complete the training.
IS_PRETRAIN = True
data_dir = 'D:\\data\\1-10\\data_fw3'
CLUSTER = 2
folder_clustermaps = 'D:\\data\\1-10\\cluster_maps_5'


# %%   Training
def train():

    pre_trained_weights = './vgg16_weights/vgg16.npy'
    # ckpt_path = 'C:\\Users\\xy31\\PycharmProjects\\VGG\\logs_cluster2\\train\\model_0_8000.ckpt-8000'
    train_log_dir = './/logs//train//'
    val_log_dir = './/logs//val//'
    feature_dict = input_data.get_feature_dict('D://data//1-10//data.csv', '5yrs')

    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))

    logits = VGG.VGG16_nobn(x, N_CLASSES, IS_PRETRAIN)

    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, ckpt_path)

        # load the parameter file, assign the parameters, skip the specific layers
        # print('**  Loading pre-trained weights  **')
        # tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

        shuffled_list_train = input_data.get_full_list(data_type='train', cluster=CLUSTER,
                                                       folder=data_dir, folder_clustermaps=folder_clustermaps)
        shuffled_list_val = input_data.get_full_list(data_type='val', cluster=CLUSTER,
                                                     folder=data_dir, folder_clustermaps=folder_clustermaps)

        shuffled_list_val = np.hstack((shuffled_list_val, shuffled_list_val))
        shuffled_list_val = np.hstack((shuffled_list_val, shuffled_list_val))
        shuffled_list_val = np.hstack((shuffled_list_val, shuffled_list_val))

        try:

            for epoch in np.arange(MAX_EPOCH):
                np.random.shuffle(shuffled_list_train)
                np.random.shuffle(shuffled_list_val)

                max_step = int(len(shuffled_list_train) / BATCH_SIZE)
                for step in np.arange(max_step):
                    try:
                        tra_image_batch, tra_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                                               batch_size=BATCH_SIZE,
                                                                                               step=step,
                                                                                               feature_dict=feature_dict,
                                                                                               n_classes=N_CLASSES,
                                                                                               name_list=shuffled_list_train)
                    except:
                        step += 1
                        try:
                            tra_image_batch, tra_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                                          batch_size=BATCH_SIZE,
                                                                                          step=step,
                                                                                          feature_dict=feature_dict,
                                                                                          n_classes=N_CLASSES,
                                                                                          name_list=shuffled_list_train)
                        except:
                            step += 1
                            tra_image_batch, tra_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                                          batch_size=BATCH_SIZE,
                                                                                          step=step,
                                                                                          feature_dict=feature_dict,
                                                                                          n_classes=N_CLASSES,
                                                                                          name_list=shuffled_list_train)
                    if coord.should_stop():
                        break

                    tra_labels = sess.run(tra_label_batch)
                    tra_images = tra_image_batch
                    _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                                    feed_dict={x: tra_images, y_: tra_labels})
                    # print(sess.run(logits, feed_dict={x: tra_images}))
                    if step % 10 == 0:
                        print('Epoch: %d (MAX_EPOCH = %d), Step: %d (MAX_Step = %d), loss: %.4f, accuracy: %.4f%%' % (epoch,
                                                                                                                    MAX_EPOCH,
                                                                                                                    step,
                                                                                                                    max_step,
                                                                                                                    tra_loss,
                                                                                                                    tra_acc))

                        summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                        tra_summary_writer.add_summary(summary_str, step)

                    if step % 50 == 0:
                        val_image_batch, val_label_batch = input_data.read_local_data(data_dir=data_dir,
                                                                                               batch_size=BATCH_SIZE,
                                                                                               step=step/50,
                                                                                               feature_dict=feature_dict,
                                                                                               n_classes=N_CLASSES,
                                                                                               name_list=shuffled_list_val)
                        val_labels = sess.run(val_label_batch)
                        val_images = val_image_batch
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={x: val_images, y_: val_labels})
                        print('**  Epoch: %d, Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (epoch,
                                                                                                      step,
                                                                                                      val_loss,
                                                                                                      val_acc))

                        summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                        val_summary_writer.add_summary(summary_str, step)

                        logits_array = sess.run(logits, feed_dict={x: tra_images})
                        labels_array = sess.run(y_, feed_dict={y_: tra_labels})
                        logits_array = np.around(logits_array, decimals=3)
                        print('==========TRAAIN==========')
                        print(np.hstack((logits_array, labels_array)))

                        logits_array = sess.run(logits, feed_dict={x: val_images})
                        labels_array = sess.run(y_, feed_dict={y_: val_labels})
                        logits_array = np.around(logits_array, decimals=3)
                        print('=========VALIDATE=========')
                        print(np.hstack((logits_array, labels_array)))

                        # tools.print_all_variables()
                        # print()

                        if step % 2000 == 0:
                            checkpoint_path = os.path.join(train_log_dir, 'model_' + str(epoch) + '_' + str(step) + '.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


def whole_validate(ckpt_path):
    feature_dict = input_data.get_feature_dict('D:/data/data.csv', 'CRC death')
    log_dir = 'C:\\Users\\xy31\\PycharmProjects\\VGG\\logs\\train\\'
    val_path = 'D:\\data\\1-16\\whole_val\\'
    collections = os.listdir(val_path)
    collection_correct = 0

    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    labels = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))
    logits = VGG.VGG16(x, N_CLASSES, keep_prob=1, is_pretrain=False)
    saver = tf.train.Saver(tf.global_variables())
    correct = tools.num_correct_prediction(logits, labels)

    with tf.Session() as sess:
        print('Reading check points')
        saver.restore(sess, ckpt_path)
        print('Loading successful')

        for collection in collections:
            # print('validating ' + collection)
            collection_path = os.path.join(val_path, collection)
            tile_list = os.listdir(collection_path)
            n_tiles = len(tile_list)
            num_step = int(math.floor(n_tiles / BATCH_SIZE))
            num_sample = num_step * BATCH_SIZE
            num_correct = 0
            for step in range(num_step):
                val_image_batch, val_label_batch = input_data.read_local_data_CRCdeath(data_dir=data_dir,
                                                                                       is_train=False,
                                                                                       batch_size=BATCH_SIZE,
                                                                                       step=step,
                                                                                       feature_dict=feature_dict,
                                                                                       n_classes=N_CLASSES,
                                                                                       name_list=tile_list)
                val_labels = sess.run(val_label_batch)
                val_images = val_image_batch
                batch_correct = sess.run(correct, feed_dict={x: val_images, labels: val_labels})
                num_correct += np.sum(batch_correct)

            print(collection + ' accuracy: %.2f%%' % (100*num_correct / num_sample))
            if num_correct >= num_sample/2:
                collection_correct += 1

        collection_accuracy = collection_correct / len(collections)

        print('Total testing collections: %.2f%%' % (100 * collection_accuracy))


def get_cnn_output(data_type, ckpt_path):
    feature_dict = input_data.get_feature_dict('D:/data/data.csv', 'CRC death')
    log_dir = 'C:\\Users\\xy31\\PycharmProjects\\VGG\\logs\\train\\'
    train_path = os.path.join(data_dir, data_type)
    list_alltiles = os.listdir(train_path)
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    labels = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))
    logits = VGG.VGG16(x, N_CLASSES, keep_prob=1, is_pretrain=False)
    saver = tf.train.Saver(tf.global_variables())
    correct = tools.num_correct_prediction(logits, labels)

    data_dict = {}
    output_dict = {}

    with tf.Session() as sess:
        print('Reading check points')
        saver.restore(sess, ckpt_path)
        print('Loading successful')

        for tile_name in list_alltiles:
            image_index = tile_name[12:16]
            aug_index = tile_name[23]
            if image_index not in data_dict.keys():
                data_dict[image_index] = dict()
                data_dict[image_index][aug_index] = [tile_name]
            elif aug_index not in data_dict[image_index].keys():
                data_dict[image_index][aug_index] = [tile_name]
            else:
                data_dict[image_index][aug_index].append(tile_name)

            if image_index not in output_dict:
                output_dict[image_index] = {}

            if aug_index not in output_dict[image_index]:
                output_dict[image_index][aug_index] = []

        progress_index = 1

        for key_image in data_dict.keys():
            aug_dict = data_dict[key_image]
            for key_aug in aug_dict:
                tile_names = aug_dict[key_aug]
                n_padded = 0
                while len(tile_names) % BATCH_SIZE != 0:
                    tile_names.append(tile_names[0])
                    n_padded += 1
                n_tiles = len(tile_names)
                num_step = int(n_tiles / BATCH_SIZE)
                for step in range(num_step):
                    batch_tiles = tile_names[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
                    if data_type == 'train':
                        is_train = True
                    elif data_type == 'validate':
                        is_train = False
                    else:
                        return 1
                    val_image_batch, val_label_batch = input_data.read_local_data_CRCdeath(data_dir=data_dir,
                                                                                           is_train=is_train,
                                                                                           batch_size=BATCH_SIZE,
                                                                                           step=step,
                                                                                           feature_dict=feature_dict,
                                                                                           n_classes=N_CLASSES,
                                                                                           name_list=tile_names)
                    val_images = val_image_batch
                    logits_array = sess.run(logits, feed_dict={x: val_images})
                    logits_array = logits_array.tolist()
                    if step != num_step - 1:
                        for i in range(BATCH_SIZE):
                            output_dict[key_image][key_aug].append(logits_array[i])
                    else:
                        for i in range(BATCH_SIZE - n_padded):
                            output_dict[key_image][key_aug].append(logits_array[i])

            print('finished %d file(s) of %d files ' % (progress_index, len(data_dict.keys())))
            progress_index += 1

        with open('cnn_output_' + data_type + '.json', 'w') as file_output:
            file_output.write(json.dumps(output_dict, indent=4))
        # json_ = json.dumps(output_dict)
        # f = open('output.json', 'w')
        # f.write(json_)
        # f.close()


def tile_validation(ckpt_path, cluster, folder_clustermaps):
    BATCH_SIZE = 48
    feature_dict = input_data.get_feature_dict('D://data//1-10//data.csv', '5yrs')

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
        list_val = input_data.get_full_list(data_name=['2018_06_01__5836'] , cluster=cluster,
                                            folder=data_dir, folder_clustermaps=folder_clustermaps)
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


def get_patch_prediction(ckpt_path, cluster, folder_clustermaps):
    feature_dict = input_data.get_feature_dict('D://data//1-10//data.csv', '5yrs')
    list_image = os.listdir(data_dir)
    # shuffle(list_image)

    output_dict = 'D:\\data\\1-10\\predictions\\' + os.path.basename(ckpt_path)

    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)

    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))

    logits = VGG.VGG16(x, N_CLASSES, keep_prob=1, is_pretrain=False)

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
        for image in list_image[0:1]:
            image = '2018_06_01__5801'
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




