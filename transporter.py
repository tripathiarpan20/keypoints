import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

from generate_lus_data import *

from data_augments import TpsAndRotate, nop
from keypoints.models import transporter
from utils import ResultsLogger
from apex import amp
from keypoints.ds import datasets as ds
from config import config

import argparse
import sys

'''
parser = argparse.ArgumentParser(description='LUS keypoint network pytorch-lightning parallel')
parser.add_argument('--lr', type=float, default=1e-4, help='') #
parser.add_argument('--max_epochs', type=int, default=50, help='') #
parser.add_argument('--sample_rate', type=int, default=4, help='') #
parser.add_argument('--batch_size', type=int, default=32, help='') #
parser.add_argument('--num_workers', type=int, default=1, help='') #
#parser.add_argument('--metric', type=str, default='mse', help='') #
parser.add_argument('--name', type=str, default='out', help='') #
parser.add_argument('--data_root', type=str, default='UltrasoundVideoSummarization/', help='') #
parser.add_argument('--LUS_num_chan', type=int, default=10, help='') #
parser.add_argument('--LUS_num_keypoints', type=int, default=10, help='') #
#parser.add_argument('--vq_path', type=str, default='VQVAE_unnorm_trained.pth', help='')
parser.add_argument('--htmaplam', type=float, default=0.1, help='') #
parser.add_argument('--device', type=str, default='cuda', help='') #

args = parser.parse_args()
'''
sys.stdout = open('stdout_TPRv2_' + args.name + '.txt', 'w')
sys.stderr = open('stderr_TPRv2_' + args.name + '.txt', 'w')


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)
    run_dir = f'data/models/transporter/{args.model_type}/run_{args.run_id}'

    """ logging """
    display = ResultsLogger(run_dir=run_dir,
                            num_keypoints=args.model_keypoints,
                            title='Results',
                            visuals=args.display,
                            image_capture_freq=args.display_freq,
                            kp_rows=args.display_kp_rows,
                            comment=args.comment)
    display.header(args)

    """ dataset """
    '''
    datapack = ds.datasets[args.dataset]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.data_root)
    pin_memory = False if args.device == 'cpu' else True
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
    '''
    ROOT = args.data_root
    US_train = USDataset(ROOT + "train/", train=True, sample_rate = args.sample_rate)
    US_test_val = USDataset(ROOT + "val/", train=False, sample_rate = args.sample_rate)
    
    dataset = {}
    dataset["train"], dataset["val"], dataset["test"] = US_train, US_test_val, US_test_val

    train_l = DataLoader(dataset["train"], batch_size=args.batch_size, pin_memory = True, num_workers = args.num_workers)
    test_l = DataLoader(dataset["val"], batch_size=args.batch_size, pin_memory =True,num_workers = args.num_workers )
    
    
    """ data augmentation """
    '''
    if args.data_aug_type == 'tps_and_rotate':
        augment = TpsAndRotate(args.data_aug_tps_cntl_pts, args.data_aug_tps_variance, args.data_aug_max_rotate)
    else:  
        augment = nop
    '''
    """ model """
    transporter_net = transporter.make(args.model_type, args.LUS_num_chan, args.model_z_channels,
                                       args.LUS_num_keypoints, load=args.load, sigma = args.htmaplam).to(args.device)

    """ optimizer """
    optim = Adam(transporter_net.parameters(), lr=args.lr)

    """ apex mixed precision """
    if args.device != 'cpu':
        amp.initialize(transporter_net, optim, opt_level=args.opt_level)

    """ loss function """


    def l2_reconstruction_loss(x, x_, loss_mask=None):
        loss = (x - x_) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
        return torch.mean(loss)


    criterion = l2_reconstruction_loss


    def to_device(data, device):
        return tuple([x.to(device) for x in data])


    for epoch in range(1, args.max_epochs + 1):

        if not args.demo:
            """ training """
            batch = tqdm(train_l, total=len(train) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                #x, x_, loss_mask = augment(*data)
                x, x_ = data #DONE
                
                optim.zero_grad()
                x_t, z, k, m, p, heatmap, mask_xs, mask_xt = transporter_net(x, x_)

                loss = criterion(x_t, x_)

                if args.device != 'cpu':
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()

                if i % args.checkpoint_freq == 0:
                    transporter_net.save(run_dir + '/checkpoint')

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, 
                            #loss_mask,
                            type='train', depth=20, mask_xs=mask_xs, mask_xt=mask_xt)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                #x, x_, loss_mask = augment(*data)
                x, x_ = data #DONE
                
                x_t, z, k, m, p, heatmap, mask_xs, mask_xt = transporter_net(x, x_)
                loss = criterion(x_t, x_, loss_mask)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask,
                            type='test', depth=20, mask_xs=mask_xs, mask_xt=mask_xt)

            ave_loss, best_loss = display.end_epoch(epoch, optim)

            """ save if model improved """
            if ave_loss <= best_loss and not args.demo:
                transporter_net.save(run_dir + '/best')
