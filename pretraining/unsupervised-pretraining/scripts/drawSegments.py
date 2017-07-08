#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
from pathlib import Path

def main(vid, seg):
    p = Path.cwd()
    vid = p.glob(vid)
    seg = p.glob(seg)
    for v,s in zip(vid, seg):
        out = s.parts[-1]
        v = imread(v)
        s = imread(s)
        smap = np.copy(s)
        smap[:] = 0
        smap[np.where(s > 70)] = 125
        smap[np.where((s > 40) & (s <= 70))] = 255
        plt.figure()
        plt.imshow(v)
        plt.imshow(smap, cmap='viridis', alpha=0.7)
        plt.savefig(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_glob', type=str)
    parser.add_argument('segment_glob', type=str)
    args = parser.parse_args()
    main(args.video_glob, args.segment_glob)

