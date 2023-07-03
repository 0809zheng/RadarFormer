#!pip3 install -U -r requirements.txt
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Datasets
from blocks import build




# Training settings
parser = argparse.ArgumentParser(description='Model Hyperparameters')
parser.add_argument('--device', default='cpu',
                    help='device to use for training / testing')
parser.add_argument('--input_dim', default=640, type=int,
                    help="Size of the imput echos (dimension of the fast time)")

# Transformer
parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--length_seq', default=32, type=int,
                    help="Length of the Sequence")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Task
parser.add_argument('--num_joint', default=14, type=int,
                    help="Number of predefined human skeleton keypoints")
parser.add_argument('--num_action', default=8, type=int,
                    help="Number of predefined human activity categories")
parser.add_argument('--num_identity', default=10, type=int,
                    help="Number of predefined human identity categories")

# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_joint', default=1, type=float,
                    help="L1 loss coefficient of joint in the matching cost")
parser.add_argument('--set_cost_action', default=1, type=float,
                    help="CE loss coefficient of activity in the matching cost")
parser.add_argument('--set_cost_identity', default=1, type=float,
                    help="CE loss coefficient of identity in the matching cost")

# * Loss coefficients
parser.add_argument('--ce_loss_coef', default=1, type=float)
parser.add_argument('--joint_loss_coef', default=1, type=float)
parser.add_argument('--action_loss_coef', default=1, type=float)
parser.add_argument('--identity_loss_coef', default=1, type=float)

# Learning Hyperparameters
parser.add_argument('--random_seed', type=int, default=42, help="set random seed")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate at begin")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay term")
parser.add_argument('--min_lr', type=float, default=1e-6, help="the minimum learning rate")
parser.add_argument('--lr_decay', type=int, default=400, help="learning rate decay epochs")
parser.add_argument('--epochs', type=int, default=500, help="training epochs")
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Datasets Hyperparameters
parser.add_argument('--batch_size', type=int, default=2, help="actual batchsize")
parser.add_argument('--shuffle', type=bool, default=True, help="shuffle train and test datasets")

# Save and Load Path
parser.add_argument('--pretrained', type=bool, default=False, help="use pre-trained model")
parser.add_argument('--pretrain_path', type=str, default='./model/pretrained.pth', help="pre-trained model saved path")
parser.add_argument('--save_path', type=str, default='./log/', help='Location to save checkpoint models')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')



args = parser.parse_args()
print(args)



def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)


def checkpoint(net, epoch):
    model_out_path = args.save_path + "model_epoch_{}.pth".format(epoch)
    torch.save(net.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))
    

if __name__ == '__main__':
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    train_loss, batch_loss = [], []
    
    model, criterion = build(args)
    model = nn.DataParallel(model)
    # print_network(model)
    

    if args.pretrained:
        if os.path.exists(args.pretrain_path):
            model.load_state_dict(torch.load(args.pretrain_path, map_location=lambda storage, loc: storage))
            print('Pre-trained model is loaded.')
    model = model.to(args.device)
    criterion = criterion.to(args.device)
            
    train_dataset = Datasets(args.num_queries)
    train_loader = DataLoader(dataset = train_dataset, shuffle = args.shuffle,
                              batch_size = args.batch_size, num_workers = 0)
    
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    criterion.train()
    
    start = time.time()
    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch))
        epoch_start = time.time()
        for step, (curr_echo, hist_echo, label, _) in enumerate(train_loader):
            curr_echo = curr_echo.to(args.device)
            hist_echo = hist_echo.to(args.device)
            label = label.to(args.device)
            target = {
                'conf': label[:, :, 0], 'action': label[:, :, 1],
                'identity': label[:, :, 2], 'joint': label[:, :, 3:],
                }
            for k in target.keys():
                target[k] = target[k].to(args.device)
            outputs = model(curr_echo, hist_echo)   
            loss_dict = criterion(outputs, target)
            # print(loss_dict)
            losses = loss_dict['loss_ce']*args.ce_loss_coef
            losses += loss_dict['loss_joint']*args.joint_loss_coef
            losses += loss_dict['loss_action']*args.action_loss_coef
            losses += loss_dict['loss_identity']*args.identity_loss_coef 
            losses += loss_dict['loss_triplet']*args.identity_loss_coef 
    
            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()
    
            batch_loss.append(losses.item())
            print('Batch loss = {:.6f}'.format(losses.item()))
            
        print('Training loss = {:.6f}'.format(np.mean(batch_loss)))
        train_loss.append(np.mean(batch_loss))
        epoch_time = time.time() - epoch_start
        print("Epoch Time elapsed: {:.2f} seconds".format(epoch_time))


        
    if (epoch+1) % (args.snapshots) == 0:
        checkpoint(model, epoch)
         
    if (epoch+1) % (args.lr_decay) == 0:
        for param_group in optimizer.param_groups:
            if param_group['lr'] > args.min_lr:
                param_group['lr'] *= 0.5
        print('Learning rate decay: lr={}\n'.format(optimizer.param_groups[0]['lr']))
    
    end_time = time.time() - start
    print("Finished. Time elapsed: {:.2f} seconds".format(end_time))
