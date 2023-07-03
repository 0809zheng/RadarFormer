from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import torch
import os


class Datasets(Dataset):
    def __init__(self, num_target):
        initDataDir = "datas/"
        self.all_data = []
        dp, _, _ = os.walk(initDataDir)
        for inputDir in dp[1]:
            files = os.listdir(initDataDir + inputDir +'/')
            signal = np.load(initDataDir+inputDir+'.npy', allow_pickle=True).item()
            
            for file in files:
                idx = file.split('.')[0]
                label = torch.tensor(signal[int(idx)]).float().squeeze(0)
                
                if label.shape[0] != num_target:
                    label = torch.vstack((label, torch.zeros((num_target-label.shape[0],label.shape[1]))))
                
                data = scio.loadmat(initDataDir + inputDir+ '/' + file)
                curr = data['CurrentEcho']
                hist = data['HistoryEcho']
                
                curr = torch.tensor(curr).float()
                hist = torch.tensor(hist).float()
    
                self.all_data.append([curr, hist, label, file])

    def __getitem__(self, index):
        return self.all_data[index][0], self.all_data[index][1], self.all_data[index][2], self.all_data[index][3]

    def __len__(self):
        return len(self.all_data)
 

class TestDatasets(Dataset):
    def __init__(self):
        initDataDir = "datas/test/"
        self.all_data = []
        dp, _, _ = os.walk(initDataDir)
        for inputDir in dp[1]:
            files = os.listdir(initDataDir + inputDir +'/')
            
            for file in files:
                data = scio.loadmat(initDataDir + inputDir+ '/' + file)
                curr = data['CurrentEcho']
                hist = data['HistoryEcho']
                
                curr = torch.tensor(curr).float()
                hist = torch.tensor(hist).float()
    
                self.all_data.append([curr, hist, file])

    def __getitem__(self, index):
        return self.all_data[index][0], self.all_data[index][1], self.all_data[index][2]

    def __len__(self):
        return len(self.all_data)
    
if __name__ == '__main__':           
    train_dataset = Datasets(num_target=100)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True)
    print(len(train_loader))
    
    for step, (curr, hist, label, file_name) in enumerate(train_loader):
        print(curr.shape)
        print(hist.shape)
        print(label.shape)
        
        target = {
            'conf': label[:, :, 0], 'action': label[:, :, 1],
            'identity': label[:, :, 2], 'joint': label[:, :, 3:],
            }
        # print(target)
        # break