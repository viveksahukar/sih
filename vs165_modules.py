# Load required libraries

import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import pydicom
import PIL
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import tqdm
import cv2
import png
import pylab
import math

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from pydicom import dcmread, read_file

from collections import Counter

import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import sklearn.metrics as metrics

from glob import glob
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import models
import torch.nn as nn
#from torchsummary import summary
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn


from timeit import default_timer as timer

from fastai.basics import *
from fastai.medical.imaging import *
from fastai.vision.all import *

from functools import partial

train_on_gpu = cuda.is_available()

def target_slice(col, targ_pct):
    """Select the most appropriate slice in a given series using DICOM attributes SliceLocation and InstanceNumber"""
    fns = [f for f in os.listdir(col)]
    try:
        slice_num = lambda x: pydicom.dcmread(col + '/' + x, stop_before_pixels=True).SliceLocation
    except AttributeError:
        print('SliceLocation Attribute not found')
    else:
        slice_num = lambda x: pydicom.dcmread(col + '/' + x, stop_before_pixels=True).InstanceNumber
    finally:
        fns = list(sorted(fns, key=slice_num))
        img_ct = len(fns)
        idx = math.floor(img_ct * targ_pct)
        targ_fn = fns[idx]
        return targ_fn

    
class DicomDataset(Dataset):
    "Custom Pytorch Dataset class for feeding directly DICOM data into CNN."
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.annotations = self.df['label']

        
    def __len__(self):
        return len(self.annotations) # should be 476 for axial
    
    def __getitem__(self, index):
        img_path = self.df.loc[index, 'full_fpath']
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array 
        img_min = np.percentile(img, 2.5)
        img_max = np.percentile(img, 97.5)
        img_clipped = np.clip(img, img_min, img_max)
        img_norm = (img_clipped - img_min) / (img_max - img_min)
        img_resize = cv2.resize(img_norm, (224, 224))
        img_3d = np.repeat(img_resize[..., np.newaxis], 3, -1)
        img_tensor = torch.from_numpy(img_3d).permute(2, 0, 1)
        img_tensor = img_tensor.float()
        
        pid = self.df.loc[index, 'PatientID']
       
        y_label = torch.tensor(int(self.df.iloc[index, 2]))
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return(img_tensor, y_label, pid)
    

def model_train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=10,
          print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target, pid) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')


        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target, pid in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history



def loss_accuracy(history, slice):
    """Plot loss accuracy curves for training & validation sets"""
    plt.figure(figsize=plt.figaspect(0.5))

    plt.subplot(1, 2, 1)
    for c in ['train_loss', 'valid_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title(f'Training and Validation Losses - {slice}')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    for c in ['train_acc', 'valid_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title(f'Training and Validation Accuracy - {slice}')
    plt.tight_layout()
    
    plt.show()
    


def model_eval(model, valid_loader):
    """Put model in evaluation mode and test on validation and test set"""
    with torch.no_grad():
        # Set to evaluation mode
        model.eval() # change here
        patient_id = []
        y_true = []
        prob = []
        score = []
        y_pred = []

        # Validation loop
        for data, target, pid in valid_loader: # change here
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass - calculate prediction probability
            score = torch.exp(model(data))
            score = score.cpu()
            target = target.cpu()

            # Predict label for the slice
            _, pred = torch.max(score, dim=1)
            y_pred = pred.cpu()

            y_true.append(int(target))
            prob.append(float(score[:, 1]))
            patient_id.append(''.join(pid))
#             score.append(score)
#             pred.append(int(pred))



        return pd.DataFrame(list(zip(patient_id, y_true, prob)), 
                               columns=['pid', 'y_true', 'prob'])



def roc_pr(df, slice, data):
    """Calculate the fpr and tpr for all thresholds of the classification and plot ROC curve"""
    y_test = df['y_true']
    preds = df['prob']

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, preds)
    area = metrics.auc(recall, precision)

    plt.figure(figsize=plt.figaspect(0.5))

    plt.subplot(1, 2, 1)
    plt.title(f'ROC Curve - {slice} - {data}')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    plt.title(f'PR Curve - {slice} - {data}')
    plt.plot(recall, precision, 'b--', label = 'Area under PR Curve = %0.2f' % area)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.tight_layout()

    plt.show()
  
    
def df_pred(pred_ax, pred_cor_1, pred_cor_2):
    """Returns dataframe of predicted probability scores for each slice for a given set"""
    pred_cor = pred_cor_1.merge(pred_cor_2, how='inner', on='pid')
    pred_cor.rename(columns={'prob_x':'p_cor_1', 'prob_y':'p_cor_2'}, inplace=True)
    pred_cor.drop(['y_true_x', 'y_true_y'], axis=1, inplace=True)

    pred = pred_ax.merge(pred_cor, how='inner', on='pid')
    pred.rename(columns={'prob':'p_ax'}, inplace=True)
    return pred


def check_duplicate(df1, df2):
    """Check if any duplicate PatientID is present across given 2 dataframes"""
    df_check = df1.merge(df2, how='outer', on='PatientID', indicator='True')
    df_check = df_check[df_check['True'] == 'both']
    print("No Duplicate PatientID found.") if len(df_check.index)==0 else print("Duplicate PatientID found.") 
