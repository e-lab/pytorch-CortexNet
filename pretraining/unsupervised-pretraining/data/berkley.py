import pathlib
import os
from random import shuffle
import time

import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class BatchSampler(data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        vids = list(self.dataset.videos)
        shuffle(vids)
        vids = iter(vids)
        # the list of videos from which frames will be served
        batch_videos = [None]*self.batch_size
        for _ in range(len(self)):
            batch_frames = [None]*self.batch_size
            reload_idx = []
            for i, vid in enumerate(batch_videos):
                if vid is None:
                    reload_idx.append(i)
                    continue
                try:
                    batch_frames[i] = next(vid)
                except StopIteration:
                    reload_idx.append(i)
            for i in reload_idx:
                # none of these next should cause stop iter
                batch_videos[i] = iter(self.dataset.vid_index[next(vids)])
                batch_frames[i] = next(batch_videos[i])
            # This can be modified to yield batch_frames
            # once batch_sampler is added to DataLoader
            for idx in batch_frames:
                yield idx

    def __len__(self):
        return len(self.dataset)#//self.batch_size

class UnsupervisedVideo(data.Dataset):

    def __init__(self, root, transform=transforms.ToTensor(), split=None):
        start_time = time.time()
        root = pathlib.Path(root)
        frame_path = root / 'UnsupVideo_Frames'
        seg_path = root / 'UnsupVideo_Segments'
        if not (frame_path.exists() and seg_path.exists()):
            print("Dataset Error")
            exit(-1)

        frame_path = list(os.walk(frame_path.as_posix()))
        seg_path = list(os.walk(seg_path.as_posix()))

        if not split is None:
            if split > 0:
                start = 0
                end = int(len(frame_path)*split)
            if split < 0:
                split = 1 + split
                start = int(len(frame_path)*split)
                end = -1
            frame_path = frame_path[start:end]
            seg_path = seg_path[start:end]

        vid_index = {}
        data = []
        videos = []
        for (fbase,_,fnames), (sbase,_,snames) in zip(frame_path, seg_path):
            for cf, nf, ns in zip(fnames[:-1], fnames[1:], snames[1:]):
                fvid = cf.split('/')[-1].split('_')[0]
                svid = ns.split('/')[-1].split('_')[0]
                if not fvid == svid:
                    continue
                if fvid not in vid_index:
                    vid_index[fvid] = []
                    videos.append(fvid)
                vid_index[fvid].append(len(data))
                data.append((os.path.join(fbase, cf),
                             os.path.join(fbase, nf),
                             os.path.join(sbase, ns),
                             fvid))

        self.vid_index = vid_index
        self.data = data
        self.videos = videos
        self.transform = transform

        print('Dataset init took %d secs'%(time.time() - start_time))

    def __getitem__(self, index):
        cframe, nframe, seg, vid = self.data[index]
        cframe = Image.open(cframe)
        nframe = Image.open(nframe)
        seg   = Image.open(seg)

        cframe = self.transform(cframe)
        nframe = self.transform(nframe)
        seg = self.transform(seg)

        return cframe, nframe, seg, vid

    def __len__(self):
        return len(self.data)
