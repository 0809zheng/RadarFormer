from __future__ import print_function

import cv2
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from data import TestDatasets
from blocks import build_test
from src.draw import Plotter3d


# Test settings
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
parser.add_argument('--threshold', default=0.5, type=float,
                    help="Confidence threshold for detecting targets")

# Datasets Hyperparameters
parser.add_argument('--batch_size', type=int, default=1, help="actual batchsize")
parser.add_argument('--shuffle', type=bool, default=False, help="shuffle test dataset")

# Save and Load Path
parser.add_argument('--pretrain_path', type=str, default='./model/pretrained.pth', help="pre-trained model saved path")
parser.add_argument('--save_path', type=str, default='./log/', help='Location to save checkpoint models')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')


args = parser.parse_args()
print(args)

ACTIVITY = ['stand', 'walk', 'jump', 'wave', 'stretch', 'box', 'sit', 'lie']
IDENTITY = ['id'+str(i) for i in range(10)]


if __name__ == '__main__':
    assert args.batch_size == 1, 'Only accept single sample during testing'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = TestDatasets()
    test_loader = DataLoader(dataset = test_dataset, shuffle = args.shuffle, batch_size = args.batch_size, num_workers = 0)
    
    model = build_test(args)
    # print(model)
    
    model.load_state_dict(torch.load(args.pretrain_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model = model.to(device)


    canvas_3d = np.ones((1280, 1600, 3), dtype=np.uint8)*255
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)
    
    
    with torch.no_grad():
        model.eval()
        print('Tests start.')
        for step, (curr_echo, hist_echo, name) in enumerate(test_loader):
            curr_echo = curr_echo.to(device)
            hist_echo = hist_echo.to(device)
            outputs = model(curr_echo, hist_echo)
            
            
            # target confidence
            pred_conf = outputs['pred_conf'].view(-1).detach().to("cpu").numpy()
            exist_tar = np.where(pred_conf >= args.threshold)
            
            
            # pose estimation
            pred_pose = outputs['pred_joint'].view(-1, 3*args.num_joint).detach().to("cpu").numpy()
            poses_3d = pred_pose[exist_tar]
            poses_3d = poses_3d.reshape(pred_pose.shape[0], args.num_joint, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + args.num_joint * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2)) 
            
            plotter.plot(canvas_3d, poses_3d, edges)
            cv2.imshow(canvas_3d_window_name, canvas_3d)
            
            
            # activity recognition
            pred_acts = outputs['outputs_action'].view(-1, args.num_action).detach().to("cpu").numpy()
            pred_acts = pred_acts[exist_tar]
            pred_acts = np.argmax(pred_acts, axis=-1)
            print('Activity predictions: ')
            for idx, pred in enumerate(pred_acts):
                print('Target {0}: {1}.'.format(idx, ACTIVITY[pred]), end='')
            print('\n')
            
            
            # identity recognition
            pred_ids = outputs['pred_identity'].view(-1, args.num_identity).detach().to("cpu").numpy()
            pred_ids = pred_ids[exist_tar]
            pred_ids = np.argmax(pred_ids, axis=-1)
            print('Identity predictions: ')
            for idx, pred in enumerate(pred_ids):
                print('Target {0}: {1}.'.format(idx, IDENTITY[pred]), end='')
            print('\n')

        print('Tests finish.')
        
    