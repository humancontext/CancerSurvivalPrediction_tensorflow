import os
import numpy as np
import tensorflow as tf
import VGG
import tools
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib


MEAN_RGB = np.asarray([217, 115, 210], dtype='uint8')  # all pixels are subtracted by the mean RGB value of all WSI
# dimensions of patches
IMG_W = 224
IMG_H = 224
IMG_D = 3
n_cluster = 10  # number of clusters

output_folder = 'D:\\data\\1-10\\cluster_maps_pheno_' + str(n_cluster)  # the folder for output cluster classification map
heatmap_folder = 'D:\\data\\1-10\\heat_maps_pheno_' + str(n_cluster)
input_folder = 'D:\\data\\1-10\\10_png_tiles_norm'  # the path where all patches are stored
cluster_train_folder = 'D:\\data\\1-10\\tiles_to_cluster'  # a collection of patches used to train the classifier

# configurations of CNN
BATCH_SIZE = 1
pre_trained_weights = './vgg16_weights/vgg16.npy'
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_W, IMG_H, IMG_D))
y_ = tf.placeholder(tf.int16, shape=(BATCH_SIZE, None))
logits = VGG.VGG16_no_fc(x, keep_prob=1)
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
init = tf.global_variables_initializer()
sess.run(init)

# load pretrained weights
print('**  Loading pre-trained weights  **')
tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])


# get the phenotype as the output of the CNN
def get_phenotype(img):
    img = np.asarray(img)
    img -= MEAN_RGB
    img = img.reshape((BATCH_SIZE, IMG_W, IMG_H, IMG_D))
    phenotype = sess.run(logits, feed_dict={x: img})
    phenotype = np.mean(phenotype, axis=0)
    phenotype = np.mean(phenotype, axis=0)
    phenotype = np.mean(phenotype, axis=0)
    return phenotype


# The phenotype clustering classifier is trained with this method.
def train_pheno_cluster():
    list_phenotype = []
    list_h = []
    list_w = []
    max_w = 0
    max_h = 0
    list_tiles = os.listdir(cluster_train_folder)
    index = 1
    for tile_name in list_tiles:
        print('finished %.2f%% ' % (index / len(list_tiles) * 100), end='\r', flush=True)
        index += 1
        path_tile = os.path.join(cluster_train_folder, tile_name)
        tile = io.imread(path_tile)
        # if len(tile_batch) == BATCH_SIZE:
        phenotypes = get_phenotype(tile)
        list_phenotype.append(phenotypes)
        h = int(tile_name[17:20])
        w = int(tile_name[21:24])
        list_h.append(h)
        list_w.append(w)
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    dcp = PCA(n_components=50)
    print('Fitting PCA')
    dcp.fit(list_phenotype)
    list_phenotype_reduced = dcp.transform(list_phenotype)
    joblib.dump(dcp, 'pca.pkl')

    print('Fitting K-Means')
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(list_phenotype_reduced)
    # cluster = kmeans.predict(list_phenotype_reduced)

    # print('Fitting SVM with the clusters')
    # clf = SVC(kernel='rbf')
    # clf.fit(X=list_phenotype_reduced, y=cluster)
    joblib.dump(kmeans, 'kmeans_10.pkl')


# Apply the trained classifer to all WSI to generate the cluster maps.
def save_pheno_clustermap():
    list_images = os.listdir(input_folder)
    for image in list_images:
        max_h, max_w = 0, 0
        folder_image = os.path.join(input_folder, image)
        list_tiles = os.listdir(folder_image)
        index = 1
        list_h = []
        list_w = []
        list_phenotype = []

        for tile_name in list_tiles:
            # print('Image %s: %d tiles(s) of %d tiles' % (image, index, len(list_tiles)),
            #       end='\r', flush=True)
            index += 1
            path_tile = os.path.join(folder_image, tile_name)
            tile = io.imread(path_tile)
            # if len(tile_batch) == BATCH_SIZE:
            phenotypes = get_phenotype(tile)
            list_phenotype.append(phenotypes)
            h = int(tile_name[17:20])
            w = int(tile_name[21:24])
            list_h.append(h)
            list_w.append(w)
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        print('PCA analyzing')
        dcp = joblib.load('pca.pkl')
        list_phenotype_reduced = dcp.transform(list_phenotype)

        print('Kmeans classifying')
        clf = joblib.load('kmeans_10.pkl')

        print('Saving cluster map')
        cluster_map = np.zeros((max_h + 1, max_w + 1))
        for i in range(len(list_tiles)):
            pheno = list_phenotype_reduced[i]
            h = list_h[i]
            w = list_w[i]
            cluster_map[h][w] = clf.predict(pheno.reshape(1, -1))
        path_data = os.path.join(output_folder, image)
        # cluster_map = np.asarray(cluster_map)
        np.save(path_data, cluster_map)

        ax = sns.heatmap(cluster_map, vmin=0, vmax=9)
        plt.show()


if __name__ == '__main__':
    train_pheno_cluster()
    save_pheno_clustermap()
