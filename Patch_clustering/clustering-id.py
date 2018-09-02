import os
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


n_clusters = 3  # number of clusters
output_folder = 'D:\\data\\1-10\\cluster_maps_id_' + str(n_clusters)  # the folder for output cluster classification map
input_folder = 'D:\\data\\1-10\\10_png_tiles_norm'  # the path where all patches are stored
cluster_train_folder = 'D:\\data\\1-10\\tiles_to_cluster'  # a collection of patches used to train the classifier


# The information density clustering classifier is trained with this method.
def train_id_cluster():
    list_tiles = os.listdir(cluster_train_folder)
    list_sizes = []
    for tile_name in list_tiles:
        path_tile = os.path.join(cluster_train_folder, tile_name)
        list_sizes.append(os.path.getsize(path_tile))
    list_sizes = np.asarray(list_sizes)
    list_sizes = np.reshape(list_sizes, (-1, 1))
    clf = KMeans(n_clusters=n_clusters, max_iter=500)
    clf.fit(list_sizes)
    joblib.dump(clf, 'kmeans_id_' + str(n_clusters) + '.pkl')


# Apply the trained classifer to all WSI to generate the cluster maps.
def save_id_clustermap():
    list_images = os.listdir(input_folder)
    clf = joblib.load('kmeans_id_' + str(n_clusters) + '.pkl')
    for image in list_images:
        max_h, max_w = 0, 0
        folder_image = os.path.join(input_folder, image)
        list_tiles = os.listdir(folder_image)
        index = 1
        list_h = []
        list_w = []
        list_size = []

        for tile_name in list_tiles:
            print('Image %s: %d tiles(s) of %d tiles' % (image, index, len(list_tiles)),
                  end='\r', flush=True)
            index += 1
            path_tile = os.path.join(folder_image, tile_name)
            size = os.path.getsize(path_tile)
            list_size.append(size)
            h = int(tile_name[17:20])
            w = int(tile_name[21:24])
            list_h.append(h)
            list_w.append(w)
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        list_size = np.asarray(list_size)
        list_size = np.reshape(list_size, (-1, 1))
        cluster_map = np.zeros((max_h + 1, max_w + 1))
        clusters = clf.predict(list_size)
        for i in range(len(list_size)):
            h = list_h[i]
            w = list_w[i]
            cluster_map[h][w] = clusters[i]
        path_data = os.path.join(output_folder, image)
        # path_heatmap = os.path.join(heatmap_folder, image)
        np.save(path_data, cluster_map)
        ax = sns.heatmap(cluster_map, vmin=0, vmax=2)
        plt.show()
        # plt.savefig(path_heatmap + '.png')


if __name__ == '__main__':
    train_id_cluster()
    save_id_clustermap()