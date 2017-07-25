import time
import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as f
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms

from model.CortexNet import CortexNetSeg as Model
from data.berkley import UnsupervisedVideo, collate_fn

from visdom import Visdom
vis = Visdom()

def masked_logistic_loss(pred, target, valid):
    mask = ((target < 0.4) + (target > 0.7)).float()
    target = (target > 0.7).float() - (target < 0.4).float()
    loss = torch.log(1 + torch.exp(-target*pred))
    loss = loss * mask
    loss = loss.view(loss.size()[0], -1).sum(1)
    mask = mask.view(mask.size()[0], -1).sum(1)
    for i, s in enumerate(valid):
        if not s:
            loss[i] = 0
            mask[i] = 0
    loss = loss.sum() if mask.sum().data[0] == 0 else loss.sum() / mask.sum()

    return loss

def main(args):
    # Create data loaders
    transform = transforms.Compose([
        transforms.Scale(int(np.max(args.spatial_size))),
        transforms.CenterCrop(args.spatial_size),
        transforms.ToTensor()
    ])
    # Load first 80% as training dataset
    train_dataset = UnsupervisedVideo(args.data_dir, args.seq_length, transform, split=0.8)
    # Load last 20% as val dataset
    val_dataset = UnsupervisedVideo(args.data_dir, args.seq_length, transform, split=-0.2)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        sampler = RandomSampler(train_dataset),
        num_workers = args.nworkers,
        pin_memory = args.cuda,
        collate_fn = collate_fn
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        sampler = RandomSampler(val_dataset),
        num_workers = args.nworkers,
        pin_memory = args.cuda,
        collate_fn = collate_fn
    )

    # Load model, define loss and optimizers
    model = Model(args.network_size)
    if args.pre_trained:
        model.load_state_dict(torch.load(args.pre_trained))
    optimizer = optim.SGD(params = model.parameters(),
                          lr = args.lr,
                          momentum = args.momentum,
                          weight_decay = args.weight_decay)
    gamma, step = 1, 1
    if args.lr_decay:
        gamma, step = args.lr_decay
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step, gamma)

    if args.cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    # Start training
    for epoch in range(args.epoch):
        #scheduler.step()
        if epoch > 0 and epoch % step == 0:
            args.lr = args.lr * gamma
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
        print('| {:3d} epoch | {:2.5f} lr'.format(epoch+1, args.lr))

        # Train
        train(model, train_loader, optimizer, args)

        # Validate
        validate(model, val_loader, args)

        # Save model parameters
        torch.save(model.state_dict(),
                   str(Path(args.out_dir)/('ckpt-%d'%epoch)))


def train(model, data_loader, optimizer, args):

    batch_time = 0
    total_loss = {
        'seg' : 0,
        'frame': 0
    }

    model.train()  # set model in train mode

    l1_loss = nn.L1Loss()
    if args.cuda:
        l1_loss = l1_loss.cuda()

    for batch_no, data in enumerate(data_loader):
        start = time.time()

        # reset state
        state = None 

        # accumulate loss
        loss = 0
        for cframe, nframe, seg, valid in data:

            # predict for every frame in sequence
            if args.cuda:
                cframe = cframe.cuda(async=True)
                nframe = nframe.cuda(async=True)
                seg = seg.cuda(async=True)
            pred_frame, pred_seg, state = model(V(cframe), state)

            # compute loss and accumulate
            seg_loss = masked_logistic_loss(pred_seg, V(seg), valid)
            # set invalid data to 0 loss
            for i, s in enumerate(valid):
                if not s:
                    pred_frame.data[i].copy_(nframe[i])
            frame_loss = l1_loss(pred_frame, V(nframe))
            loss += (args.alpha * seg_loss) + (args.beta * frame_loss)

            # store loss for future logging
            total_loss['seg'] += seg_loss.data[0]
            total_loss['frame'] += frame_loss.data[0]

        # propogate accumulated loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # record time for logging
        batch_time += time.time() - start

        # print loss information
        if batch_no % args.log_interval == 0:
            cframe = cframe.cpu().numpy()[0]
            nframe = nframe.cpu().numpy()[0]
            pred_frame = pred_frame.data.cpu().numpy()[0]
            seg = seg.cpu().numpy()[0][0]
            pred_seg = pred_seg.data.cpu().numpy()[0][0]
            smap = np.zeros_like(cframe)
            smap[0][seg > 0.7] = 1
            seg = smap
            smap = np.zeros_like(cframe)
            smap[0] = pred_seg
            smap[1][pred_seg > 0.7] = 1
            pred_seg = smap
            # Draw stuff
            vis.image(cframe,
                      win='cframe-batch-%d'%batch_no,
                      opts=dict(title='cframe-batch-%d'%batch_no))
            vis.image(nframe,
                      win='nframe-batch-%d'%batch_no,
                      opts=dict(title='nframe-batch-%d'%batch_no))
            vis.image(pred_frame,
                      win='predframe-batch-%d'%batch_no,
                      opts=dict(title='predframe-batch-%d'%batch_no))
            vis.image(np.abs(pred_frame-nframe),
                      win='framediff-batch-%d'%batch_no,
                      opts=dict(title='framediff-batch-%d'%batch_no))
            vis.image(seg,
                      win='seg-batch-%d'%batch_no,
                      opts=dict(title='seg-batch-%d'%batch_no))
            vis.image(pred_seg,
                      win='predseg-batch-%d'%batch_no,
                      opts=dict(title='predseg-batch-%d'%batch_no))
            # Print stuff
            avg_time = batch_time * 10e3 / args.log_interval
            total_loss['seg'] /= (args.log_interval * args.seq_length)
            total_loss['frame'] /= (args.log_interval * args.seq_length)
            print('| {:4d}/{:4d} batches| {:7.2f} ms/batch |'
                  ' {:.2e} seg_loss | {:.2e} frame_loss |'.
                  format(batch_no + 1, len(data_loader),
                         avg_time, total_loss['seg'], total_loss['frame']))
            for k in total_loss: total_loss[k] = 0
            batch_time = 0

def validate(model, data_loader, args):


    batch_time = 0
    total_loss = {
        'seg' : 0,
        'frame': 0
    }

    model.eval()  # set model in val mode

    l1_loss = nn.L1Loss()
    if args.cuda:
        l1_loss = l1_loss.cuda()

    for batch_no, data in enumerate(data_loader):
        start = time.time()

        # reset state
        state = None 

        # accumulate loss
        loss = 0
        for cframe, nframe, seg, valid in data:
            # predict for every frame in sequence
            if args.cuda:
                cframe = cframe.cuda(async=True)
                nframe = nframe.cuda(async=True)
                seg = seg.cuda(async=True)
            pred_frame, pred_seg, state = model(V(cframe), state)

            # compute loss and accumulate
            seg_loss = masked_logistic_loss(pred_seg, V(seg), valid)
            # set invalid data to 0 loss
            for i, s in enumerate(valid):
                if not s:
                    pred_frame[i].copy_(nframe[i])
            frame_loss = l1_loss(pred_frame, V(nframe))
            loss += (args.alpha * seg_loss) + (args.beta * frame_loss)

            # store loss for future logging
            total_loss['seg'] += seg_loss.data[0]
            total_loss['frame'] += frame_loss.data[0]

        # propogate accumulated loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # record time for logging
        batch_time += time.time() - start

    # print loss information
    avg_time = batch_time * 10e3 / len(data_loader)
    total_loss['seg'] /= (len(data_loader) * args.seq_length)
    total_loss['frame'] /= (len(data_loader) * args.seq_length)
    print('| val | {:4d} batches| {:7.2f} ms/batch |'
          ' {:.2e} seg_loss | {:.2e} frame_loss |'.
          format(len(data_loader), avg_time,
                 total_loss['seg'], total_loss['frame']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, dest='data_dir',
                        help='base folder of data', required=True)
    parser.add_argument('--batch-size', '-B', type=int, default=20,
                        metavar='B', help='batch size')
    parser.add_argument('--seq-length', '-T', type=int, default=5,
                        metavar='T', help='Sequence length to train on')
    parser.add_argument('--nworkers', type=int, default=1,
                        metavar='W', help='num of dataloader workers')
    parser.add_argument('--spatial-size', type=int,
                        default=(256, 256), nargs=2,
                        help='frame cropping size', metavar=('H', 'W'))
    parser.add_argument('--network-size', type=int, nargs='*', metavar='S',
                        default=(3,32,64,128), help='sizes of hidden layers')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--alpha', type=float, default=1,
                        help='seg loss multiplier')
    parser.add_argument('--beta', type=float, default=1,
                        help='frame loss multiplier')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='W', help='weight decay')
    parser.add_argument('--cuda', action='store_true',
                        default=False, help='use cuda')
    parser.add_argument('--epoch', '-E', default=10, metavar='NUM_EPOCH',
                        type=int, help='number of epochs to train for')
    parser.add_argument('--pre-trained', type=str, default='', metavar='P',
                        help='path to pre-trained model parameters')
    parser.add_argument('--lr-decay', type=float, default=None, 
                        metavar=('D', 'E'), nargs=2,
                        help='decay lr by factor D every E epochs. lr = lr*D')
    parser.add_argument('--log-interval', type=int, default=10,
                        metavar='N', help='report interval')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='directory to save model')

    args = parser.parse_args()
    main(args)
