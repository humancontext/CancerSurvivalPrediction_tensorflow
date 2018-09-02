
import os
import os.path
import numpy as np
import tensorflow as tf
import input_data
import VGG
import tools

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
data_dir = 'D:\\data\\1-10\\data_fw3'  # the path where all patches are stored
pre_trained_weights = './vgg16_weights/vgg16.npy'  # the path of pretrained model


# choose clustering model and number of cluster
def patch_train(cluster, folder_clustermaps):
    '''
    :param cluster: the cluster to perform patch-wise classify
    :param folder_clustermaps: folder of cluster maps
    '''
    train_log_dir = './/logs//train//'
    val_log_dir = './/logs//val//'
    feature_dict = input_data.get_feature_dict('D://data//1-10//data.csv', feature_to_classify)

    # setup of VGG16-like CNN
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, N_CLASSES))
    logits = VGG.VGG16_nobn(x, N_CLASSES, TRAINABLE)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if IS_FINETUNING:
            # load the parameter file, assign the parameters, skip the specific layers
            print('**  Loading pre-trained weights  **')
            tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
        shuffled_list_train = input_data.get_full_list(data_type='train', cluster=cluster,
                                                       folder=data_dir, folder_clustermaps=folder_clustermaps)
        shuffled_list_val = input_data.get_full_list(data_type='val', cluster=cluster,
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
                    if step % 10 == 0:
                        print(
                            'Epoch: %d (MAX_EPOCH = %d), Step: %d (MAX_Step = %d), loss: %.4f, accuracy: %.4f%%'
                            % (
                                epoch,
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
                                                                                      step=step / 50,
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

                        # logits_array = sess.run(logits, feed_dict={x: tra_images})
                        # labels_array = sess.run(y_, feed_dict={y_: tra_labels})
                        # logits_array = np.around(logits_array, decimals=3)
                        # print('==========TRAAIN==========')
                        # print(np.hstack((logits_array, labels_array)))
                        #
                        # logits_array = sess.run(logits, feed_dict={x: val_images})
                        # labels_array = sess.run(y_, feed_dict={y_: val_labels})
                        # logits_array = np.around(logits_array, decimals=3)
                        # print('=========VALIDATE=========')
                        # print(np.hstack((logits_array, labels_array)))

                        if step % 2000 == 0:
                            checkpoint_path = os.path.join(train_log_dir,
                                                           'model_' + str(epoch) + '_' + str(step) + '.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    folder_clustermaps = 'D:\\data\\1-10\\cluster_maps_5'
    cluster = 2
    patch_train(cluster=cluster, folder_clustermaps=folder_clustermaps)