from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from PIL import Image

from p2vparser import p2vParser

im = Image.open('/home/ron/studies/project/PatchMatch/woman.png')
patches = p2vParser('/home/ron/studies/project/PatchMatch/woman-181117.p2v', h=im.size[1]-31, w=im.size[0]-31)
im_shape = patches.flow.shape
patches2v = patches.flow.reshape(-1, 128)
n_samples, n_features = im_shape[0]*im_shape[1], im_shape[2]
n_fitters = 200
chosen_patches=np.random.choice(range(n_samples), n_fitters)
n_neighbors = 15

# mds_embedder = manifold.MDS(n_components=2, random_state=0,
#                                       dissimilarity='euclidean')
# D = pairwise_distances(patches2v[chosen_patches])
# patches_2d_pca = mds_embedder.fit_transform(patches2v[chosen_patches])
# embedder2 = PCA(n_components=2)
# embedder2 = LocallyLinearEmbedding(n_neighbors, 2, method='hessian')
embedder2 = TSNE(n_components=2, init='pca', random_state=0)
patches_2d_pca = embedder2.fit_transform(patches2v[chosen_patches])

def plot_embedding(patches_2d, im_shape, im, chosen_patches, title=None):
    print ("Creating the plot")
    x_min, x_max = np.min(patches_2d[:]), np.max(patches_2d[:])
    patches_2d = (patches_2d - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(n_fitters):
            dist = np.sum((patches_2d[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-5:
                # don't show points that are too close
                continue
            shown_images = np.vstack((shown_images, patches_2d[i]))
            box = [chosen_patches[i]%im_shape[1], chosen_patches[i]//im_shape[1],
                   (chosen_patches[i]%im_shape[1])+32, (chosen_patches[i]//im_shape[1])+32]
            image = im.crop(box)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image, zoom=0.5, cmap=plt.cm.gray_r),
                patches_2d[i], pad=0)
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    print ('min: %f' % np.min(shown_images[:]))
    ax.set_xlim(np.min(shown_images[:])-0.05, np.max(shown_images[:]))
    ax.set_ylim(np.min(shown_images[:]-0.05), np.max(shown_images[:]+0.1))
    if title is not None:
        plt.title(title)

    fig.set_figheight(7)
    fig.set_figwidth(7)
    plt.title('2D embedding of Patch2Vec vector of patches', fontsize=16, weight='semibold')
    fig.savefig('patch2vec-2d.png', dpi=120, bbox_inches='tight')

plot_embedding(patches_2d_pca, im_shape, im, chosen_patches)
