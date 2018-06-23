import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from bmpnnfparser import BmpNNFParser

NNF1_PATH = os.path.expanduser('~/ann_l2.bmp')
NNF2_PATH = os.path.expanduser('~/ann_nn.bmp')

def compare_l2_p2v_nnf(nnf1p, nnf2p):
    nnf1 = BmpNNFParser(nnf1p)
    nnf2 = BmpNNFParser(nnf2p)

    dist = np.zeros([nnf1.height, nnf1.width])

    for y in range(nnf1.height):
        for x in range(nnf1.width):
            nn1 = nnf1.get_pixel(x, y)
            nn2 = nnf2.get_pixel(x, y)
            dist[y][x] = np.linalg.norm(nn1 - nn2)

    imgplot = plt.imshow(dist)
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.title('Nearest neighbor difference between \noriginal PatchMatch and Patch2Vec variant')

    plt.savefig('ann_algorithms_distance.png')

def main():
    parser = argparse.ArgumentParser(description='Analize results from NNF comparisons')
    parser.add_argument('--mode', required=True, choices=['L2_vs_P2V'], dest='mode')

    p = parser.parse_args()

    if p.mode == 'L2_vs_P2V':
        compare_l2_p2v_nnf(NNF1_PATH, NNF2_PATH)

if __name__ == '__main__':
    sys.exit(main())
