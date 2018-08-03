import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from bmpnnfparser import BmpNNFParser
from floparser import FloParser

#NNF1_PATH = os.path.expanduser('~/ann_l2.bmp')
#NNF2_PATH = os.path.expanduser('~/ann_nn.bmp')

GT_FLO_PATH = os.path.expanduser(r'd:\Development\patch-flow-wrappers\dataBase\groudTruth-flow\Grove2\flow10.flo')
NNF1_PATH = os.path.expanduser(r'd:\Development\patch-flow-wrappers\dataBase\PMresults\Grove2_5000_32x32\frame10.png_frame11.png_l2.bmp')
NNF2_PATH = os.path.expanduser(r'd:\Development\patch-flow-wrappers\dataBase\PMresults\Grove2_5000_32x32\frame10.png_frame11.png_nn.bmp')

def compare_l2_p2v_nnf(nnf1p, nnf2p):
    nnf1 = BmpNNFParser(nnf1p)
    nnf2 = BmpNNFParser(nnf2p)

    dist = np.zeros([nnf1.h_, nnf1.width])
    print('first image- height {}, width {}\nsecond image- height {}, width {}'.format(nnf1.h_, nnf1.width,nnf2.h_, nnf2.width))

    for y in range(nnf1.h_):
        for x in range(nnf1.width):
            nn1 = nnf1[y][x]
            nn2 = nnf2[y][x]
            dist[y][x] = np.linalg.norm(nn1 - nn2)

    imgplot = plt.imshow(dist)
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.title('Nearest neighbor difference between \noriginal PatchMatch and Patch2Vec variant')

    plt.savefig('ann_algorithms_distance_L2_vs_P2V.png')

def compare_gt_vs_nnf(nnf1p, nnf2p, mode):
    nnf1 = FloParser(nnf1p)
    nnf2 = BmpNNFParser(nnf2p)

    print('first image- height {}, width {}\n second image- height {}, width {}'.format(nnf1.h_, nnf1.width,nnf2.h_, nnf2.width))
    dist = np.zeros([nnf1.h_, nnf1.w_])

    for y in range(nnf1.h_):
        for x in range(nnf1.w_):
            nn1 = nnf1[y][x]
            nn2 = nnf2[y][x]
            dist[y][x] = np.linalg.norm(nn1 - nn2)

    imgplot = plt.imshow(dist)
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.title('Nearest neighbor field difference between \nGround truth and predicted Nearest neighbor')

    FileName = r'ann_algorithms_distance_{}.png'.format(mode)
    plt.savefig(FileName)

def main():
    parser = argparse.ArgumentParser(description='Analize results from NNF comparisons')
    parser.add_argument('--mode', required=True, choices=['L2_vs_P2V','GT_vs_P2V','GT_vs_PM'], dest='mode')

    p = parser.parse_args()

    if p.mode == 'L2_vs_P2V':
        compare_l2_p2v_nnf(NNF1_PATH, NNF2_PATH)

    if p.mode == 'GT_vs_P2V':
        compare_gt_vs_nnf(GT_FLO_PATH, NNF2_PATH,p.mode)

    if p.mode == 'GT_vs_PM':
        compare_gt_vs_nnf(GT_FLO_PATH, NNF1_PATH,p.mode)

if __name__ == '__main__':
    sys.exit(main())
