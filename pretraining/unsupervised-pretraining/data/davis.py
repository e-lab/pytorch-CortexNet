import os
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

class Davis(data.Dataset):
    
    def __init__(self, root, seq_file, seq_length, seq_skip = 0,
                 transform = transforms.ToTensor()):
        super().__init__()
        
        root = Path(root)
        sequences = [i.strip() for i in open(seq_file).readlines()]

        self.data = []
        for seq in sequences:
            img_path = root / 'JPEGImages/480p' / seq
            seg_path = root / 'Annotations/480p' / seq
            imgs = list(os.listdir(img_path.as_posix()))
            segs = list(os.listdir(seg_path.as_posix()))
            imgs = sorted(imgs)
            segs = sorted(segs)
            imgs = [(img_path / i).as_posix() for i in imgs]
            segs = [(seg_path / i).as_posix() for i in segs]
            cframes = imgs[:-1]
            nframes = imgs[1:]
            csegs = segs[:-1]
            nsegs = segs[1:]            
            _data = []
            for i, d in enumerate(zip(cframes, nframes, csegs, nsegs)):
                if not i % (seq_skip+1) == 0: continue
                _data.append(d)
                if len(_data) == seq_length:
                    self.data.append(_data)
                    _data = []

        self.transform = transform
        
    def __getitem__(self, index):
        sequence = self.data[index]
        curr_frames = []
        next_frames = []
        curr_segs = []
        next_segs = []
        for cf, nf, cs, ns in sequence:
            cf = self.transform(Image.open(cf))
            nf = self.transform(Image.open(nf))
            cs = (self.transform(Image.open(cs)) > 0).float()
            ns = (self.transform(Image.open(ns)) > 0).float()
            curr_frames.append(cf)
            next_frames.append(nf)
            curr_segs.append(cs)
            next_segs.append(ns)

        return curr_frames, next_frames, curr_segs, next_segs
        
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate(args):
        seq_length = len(args[0][0])
        data = []
        for seq in range(seq_length):
            cframes = []
            nframes = []
            csegs = []
            nsegs = []
            for arg in args:
                f, n, cs, ns = arg
                cframes.append(f[seq].unsqueeze(0))
                nframes.append(n[seq].unsqueeze(0))
                csegs.append(cs[seq].unsqueeze(0))
                nsegs.append(ns[seq].unsqueeze(0))
            cframes = torch.cat(cframes, 0)
            nframes = torch.cat(nframes, 0)
            csegs = torch.cat(csegs, 0)
            nsegs = torch.cat(nsegs, 0)
            data.append((cframes, nframes, csegs, nsegs))
        return data
