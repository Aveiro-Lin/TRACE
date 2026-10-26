# coding: utf-8

# In[13]:


import os

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import utils.my_ecg_process as ecg
from utils.cpsc2018_utils import CPSC2018Utils
# from utils.my_tokenizer import Tokenizer
from utils.trace_tokenizer import Tokenizer

import matplotlib.pyplot as plt
from tqdm import tqdm


# In[14]:


print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(f"Current device index: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# In[15]:


sampling_rate = 500
batch_size = 32
# batch_size = 1
# seq_length = 500
# seq_length = 1500
seq_length = 2250
# patch_size = 25
patch_size = 75
# latent_ratio = 0.5
latent_ratio = 0.1
channels = 12
codebook_size = 256
residual_levels = 2
dir = f"ecg_tokenizer_level_{residual_levels}_code_{codebook_size}_len_{seq_length}_ratio_{latent_ratio}_cpsc2018"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[16]:


# # total_length = int(seq_length * (1 + latent_ratio))
# # Y = pd.read_csv('data/ptbxl_database.csv', index_col='ecg_id')
# # X = ecg.load_raw_data(Y, 'data/ptbxl/', sampling_rate, total_length)
# # np.save("data/records500.npy", X)

# total_length = int(seq_length * (1 + latent_ratio))
# data_root = 'data/'
# cpsc2018Utils = CPSC2018Utils(os.path.join(data_root, 'cpsc2018/Training_WFDB'))

# train_reference_path = os.path.join(data_root, 'TRAIN_reference_cpsc2018.csv')
# val_reference_path = os.path.join(data_root, 'VAL_reference_cpsc2018.csv')
# test_reference_path = os.path.join(data_root, 'TEST_reference_cpsc2018.csv')
# X_train = cpsc2018Utils.load_raw_data_cpsc2018(train_reference_path, sampling_rate, total_length, handle_via_trace=True)
# X_valid = cpsc2018Utils.load_raw_data_cpsc2018(val_reference_path, sampling_rate, total_length, handle_via_trace=True)
# X_test = cpsc2018Utils.load_raw_data_cpsc2018(test_reference_path, sampling_rate, total_length, handle_via_trace=True)
# # np.save("data/records500_cpsc2018_train.npy", X_train)
# # np.save("data/records500_cpsc2018_val.npy", X_valid)
# # np.save("data/records500_cpsc2018_test.npy", X_test)
# np.save("data/records2250_cpsc2018_train.npy", X_train)
# np.save("data/records2250_cpsc2018_val.npy", X_valid)
# np.save("data/records2250_cpsc2018_test.npy", X_test)


# In[17]:


# X = np.load("data/records500.npy")
# total_length = int(seq_length * (1 + latent_ratio))
# start = int((X.shape[2] - total_length)/2)
# X = X[:, :, start:start+total_length]
# Y = pd.read_csv('data/ptbxl_database.csv', index_col='ecg_id')


# In[18]:


# valid_fold = 9
# test_fold = 10
# X_train = X[np.where((Y.strat_fold != valid_fold) & (Y.strat_fold != test_fold))]
# X_valid = X[np.where(Y.strat_fold == valid_fold)]
# X_test = X[np.where(Y.strat_fold == test_fold)]

# X_train = np.load("data/records500_cpsc2018_train.npy")
# X_valid = np.load("data/records500_cpsc2018_val.npy")
# X_test = np.load("data/records500_cpsc2018_test.npy")
X_train = np.load("data/records2250_cpsc2018_train.npy")
X_valid = np.load("data/records2250_cpsc2018_val.npy")
X_test = np.load("data/records2250_cpsc2018_test.npy")
total_length = int(seq_length * (1 + latent_ratio))
start = int((X_train.shape[2] - total_length)/2)
X_train = X_train[:, :, start:start+total_length]
X_valid = X_valid[:, :, start:start+total_length]
X_test = X_test[:, :, start:start+total_length]


# In[19]:


train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32))
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# In[20]:


vq_kwargs = {'residual_levels': residual_levels}
tokenizer = Tokenizer(
    seq_length=seq_length,
    patch_size=patch_size,
    latent_ratio=latent_ratio,
    channels=channels,
    codebook_size=codebook_size,
    vq_kwargs=vq_kwargs
).to(device)
optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# In[21]:


writer = SummaryWriter(log_dir=f"runs/{dir}")


# In[22]:


def plot_signal(epoch):
    x_train = train_dataset[random.randint(0, len(train_dataset) - 1)][0].unsqueeze(0).to(device)
    x_valid = valid_dataset[random.randint(0, len(valid_dataset) - 1)][0].unsqueeze(0).to(device)
    channel = random.randint(0, channels - 1)
    with torch.no_grad():
        _, recon_train, pred_train = tokenizer(x_train[:, :, :seq_length], predict = True)
        _, recon_valid, pred_valid = tokenizer(x_valid[:, :, :seq_length], predict = True)

    def plot(x, recon, pred, i):
        title = "Train Sample Channel {channel+1}" if i == 1 else "Valid Sample Channel {channel+1}"
        plt.subplot(1, 2, i)
        x = x.cpu().detach().numpy()[0][channel]
        recon = recon.cpu().detach().numpy()[0][channel]
        pred = pred.cpu().detach().numpy()[0][channel]
        
        total_length = len(recon) + len(pred)
        time_axis = np.arange(total_length)
        
        plt.plot(time_axis[:len(x)], x[:total_length], 
                 label=f"Original Signal", color='blue', alpha=0.7, linewidth=1.5)
        
        plt.plot(time_axis[:len(recon)], recon, 
                 label=f"Reconstructed Signal", color='purple', alpha=0.7, linewidth=1.5)
        
        pred_start = len(recon)
        plt.plot(time_axis[pred_start:], pred, 
                 label=f"Predicted Signal", color='red', alpha=0.7, linewidth=1.5)
        
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("ECG Value")
        plt.legend()

    plt.figure(figsize=(20, 10))
    plot(x_train, recon_train, pred_train, 1)
    plot(x_valid, recon_valid, pred_valid, 2)
    plt.draw()
    
    img = np.array(plt.gcf().canvas.buffer_rgba())
    img = np.transpose(img, (2, 0, 1))
    writer.add_image(f"Prediction/epoch_{epoch+1}", img, epoch)
    
    plt.close()


# In[23]:


epochs = 50
start_epoch = 0
model_path = f"ckpt/cpsc2018/tokenizer/{dir}_st2.pth"


# In[24]:


if os.path.exists(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    tokenizer.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from Epoch {start_epoch+1}")


# In[ ]:


for epoch in range(start_epoch, epochs):  
    with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch", dynamic_ncols=True) as tepoch:
        recon_loss = 0
        pred_loss = 0
        train_loss = 0
        for index, batch in enumerate(tepoch):
            x = batch[0].to(device)
            optimizer.zero_grad()
            r_loss, _, pred_sequence = tokenizer(x[:, :, :seq_length], predict = True)
            p_loss = F.mse_loss(pred_sequence, x[:, :, seq_length:seq_length+pred_sequence.shape[2]])
            loss = r_loss + p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
            optimizer.step()
            recon_loss += r_loss.detach().item()
            pred_loss += p_loss.detach().item()
            train_loss += loss.detach().item()
            tepoch.set_postfix(total_loss=loss.item())

        avg_train_recon_loss = recon_loss / len(train_loader)
        avg_train_pred_loss = pred_loss / len(train_loader)
        avg_train_loss = train_loss / len(train_loader)

        tokenizer.eval()
        recon_loss = 0
        pred_loss = 0
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch[0].to(device)
                r_loss, _, pred_sequence = tokenizer(x[:, :, :seq_length], predict = True)
                p_loss = F.mse_loss(pred_sequence, x[:, :, seq_length:seq_length+pred_sequence.shape[2]])
                loss = r_loss + p_loss
                recon_loss += r_loss.detach().item()
                pred_loss += p_loss.detach().item()
                valid_loss += loss.detach().item()
        
        avg_valid_recon_loss = recon_loss / len(valid_loader)
        avg_valid_pred_loss = pred_loss / len(valid_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        writer.add_scalars('Loss/Total', {'Train': avg_train_loss, 'Valid': avg_valid_loss}, epoch+1)
        writer.add_scalars('Loss/Recon', {'Train': avg_train_recon_loss, 'Valid': avg_valid_recon_loss}, epoch+1)
        writer.add_scalars('Loss/Pred', {'Train': avg_train_pred_loss, 'Valid': avg_valid_pred_loss}, epoch+1)
        plot_signal(epoch)
        model = {
            'model_state_dict': tokenizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(model, f"ckpt/cpsc2018/tokenizer/{dir}_50_st2.pth")
        tokenizer.train()
    scheduler.step()
    
model = {
    'model_state_dict': tokenizer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(model, model_path)
