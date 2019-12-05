from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import argparse
from timeit import default_timer as timer
import time

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.init as init
from torch.nn.parallel.data_parallel import data_parallel

from models.model import *
import torchvision.models as models
from apex import amp

import albumentations
from albumentations import torch as AT

from utils.transform import *
from utils.dataset import *
from torch.utils.tensorboard import SummaryWriter
from utils.ranger import *
import utils.learning_schedules_fastai as lsf
from utils.fastai_optim import OptimWrapper
from utils.lrs_scheduler import * 
from utils.other_loss import *
from utils.lovasz_loss import *
from utils.loss_function import *
from utils.metric import *


############################################################################## define augument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model', type=str, default='efficientnet-b5', required=False, help='specify the backbone model')
parser.add_argument('--model_type', type=str, default='unet', required=False, help='specify the model')
parser.add_argument('--optimizer', type=str, default='Ranger', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmRestart', required=False, help="specify the lr scheduler")
parser.add_argument("--lr", type=int, default=4e-3, required=False, help="specify the initial learning rate for training")
parser.add_argument("--batch_size", type=int, default=4, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=4, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=15, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--train_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization", \
    required=False, help="specify the folder for training data")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Cloud/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')


############################################################################## define constant
SEED = 42
NUM_TRAIN = 5546
NUM_TEST  = 3698

NUM_TEST_POS={ #estimae only !!!! based on public data only !!!
'Fish'   : (1864, 0.5040562466197945),
'Flower' : (1508, 0.4077879935100054),
'Gravel' : (1982, 0.5359653866955111),
'Sugar'  : (2382, 0.6441319632233640), #**
}# total pos:7736  neg:7056

NUM_TRAIN_POS={
'Fish'   : (2765, 0.499),
'Flower' : (3181, 0.574),
'Gravel' : (2607, 0.470),
'Sugar'  : (1795, 0.324),
}

CLASSNAME_TO_CLASSNO = {
'Fish'   : 0,
'Flower' : 1,
'Gravel' : 2,
'Sugar'  : 3,
}

CLASSNO_TO_CLASSNAME = {v: k for k, v in CLASSNAME_TO_CLASSNO.items()}

NUM_CLASS = len(CLASSNAME_TO_CLASSNO)

DATA_DIR = '/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization'

############################################################################## seed all
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)



############################################################################## define trainsformation
def transform_train(image, label, mask, infor):

    if random.random() < 0.5:
        image = albumentations.VerticalFlip(p=1)(image=image)['image']
        mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        
        image = albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        ])(image=image)['image']
        
        image = albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5)(image=image)['image']
        # image = albumentations.Blur(blur_limit=3)(image=image)['image']

    if random.random() < 0.5:
        image = albumentations.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1)(image=image)['image']
        mask = albumentations.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1)(image=mask)['image']
        
    # if random.random() < 0.5:
    #     image = albumentations.RandomRotate90(p=1)(image=image)['image']
    #     mask = albumentations.RandomRotate90(p=1)(image=mask)['image']

    # if random.random() < 0.5:
    #     image = albumentations.Transpose(p=1)(image=image)['image']
    #     mask = albumentations.Transpose(p=1)(image=mask)['image']
    
    # if random.random() < 0.5:
    #     image = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=image)['image']
    #     mask = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=mask)['image']
    
    image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']

    return image, label, mask, infor

def transform_valid(image, label, mask, infor):
    
    image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']

    return image, label, mask, infor


# define augumentation for TTA
def transform_test(image):
    
    image_hard = image.copy()
    image_simple = image.copy()

    if random.random() < 0.5:
        image = albumentations.VerticalFlip(p=1)(image=image)['image']
        mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']
     
     
    image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']   
    # image = albumentations.Normalize(image=image)['image']

    return image_simple, image_hard


def children(m: nn.Module):
    return list(m.children())

def num_children(m: nn.Module):
    return len(children(m))

def unet_training(model_name,
                  model_type,
                  optimizer_name,
                  lr_scheduler_name,
                  lr,
                  batch_size,
                  valid_batch_size,
                  num_epoch,
                  start_epoch,
                  accumulation_steps,
                  train_data_folder, 
                  checkpoint_folder,
                  load_pretrain
                  ):
    
    COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
    COMMON_STRING += '\tset random seed\n'
    COMMON_STRING += '\t\tSEED = %d\n'%SEED

    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += '\tset cuda environment\n'
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    COMMON_STRING += '\n'
    
    if not os.path.exists(checkpoint_folder + '/' + model_type + '/' + model_name):
        os.mkdir(checkpoint_folder + '/' + model_type + '/' + model_name)
    
    log = Logger()
    log.open(checkpoint_folder + '/' + model_type + '/' + model_name + '/' + model_name + '_log_train.txt', mode='a+')
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % train_data_folder)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % checkpoint_folder)
    log.write('\n')

    
    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CloudDataset(
        data_dir = train_data_folder,
        mode     = 'train',
        csv      = ['train.csv',],
        split    = ['by_random1/train_fold_a0_5246.npy',],
        augment  = transform_train,
    )
    train_dataloader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = CloudDataset(
        data_dir = train_data_folder,
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['by_random1/valid_fold_a0_300.npy',],
        #split   = ['by_random1/valid_small_fold_a0_120.npy',],
        augment = transform_valid,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = valid_batch_size,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ############################################################################## define unet model with backbone
    MASK_WIDTH = 525
    MASK_HEIGHT = 350
    
    def get_unet_model(model_name="efficientnet-b3", IN_CHANNEL=3, NUM_CLASSES=2, WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT):
        model = model_iMet(model_name, IN_CHANNEL, NUM_CLASSES, WIDTH, HEIGHT)
        
        # Optional, for multi GPU training and inference
        # model = nn.DataParallel(model)
        return model


    ############################################################################### training parameters
    checkpoint_filename = model_type + '/' + model_name + '/' + model_name + "_" + model_type + "_unet_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################### model and optimizer
    model = get_unet_model(model_name=model_name, IN_CHANNEL=3, NUM_CLASSES=len(CLASSNAME_TO_CLASSNO), WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT)
    if (load_pretrain):
        model.load_pretrain(checkpoint_filepath)
    model = model.cuda()
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "adamonecycle":
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=1e-4, true_wd=True, bn_wd=True
        )
    elif optimizer_name == "Ranger":
        optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=1e-5)
    else:
        raise NotImplementedError
    
    if lr_scheduler_name == "adamonecycle":
        scheduler = lsf.OneCycle(optimizer, len(train_dataset) * num_epoch, lr, [0.95, 0.85], 10.0, 0.4)
        lr_scheduler_each_iter = True
    elif lr_scheduler_name == "CosineAnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)
        lr_scheduler_each_iter = False
    else:
        raise NotImplementedError

    log.write('net\n  %s\n'%(model_name))
    log.write('optimizer\n  %s\n'%(optimizer_name))
    log.write('schduler\n  %s\n'%(lr_scheduler_name))
    log.write('\n')

    # mix precision
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    ############################################################################### training
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  accumulation_steps=%d\n'%(batch_size, accumulation_steps))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    
    valid_loss = np.zeros(17,np.float32)
    train_loss = np.zeros( 6,np.float32)
    valid_metric_optimal = np.inf
    eval_step = len(train_dataloader) # or len(train_dataloader) 
    log_step = 100
    eval_count = 0
    
    # define tensorboard writer and timer
    writer = SummaryWriter()
    start_timer = timer()
    
    for epoch in range(1, num_epoch+1):
        
        # update lr and start from start_epoch  
        if (not lr_scheduler_each_iter):
            if epoch < 6:
                if epoch != 0:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2) 
            elif epoch > 5 and epoch < 7:
                optimizer.param_groups[0]['lr'] = 1e-5
            else:
                optimizer.param_groups[0]['lr'] = 5e-6
            
        if (epoch < start_epoch):
            continue
        
        log.write("Epoch%s\n" % epoch)
        log.write('\n')
        
        for param_group in optimizer.param_groups:
            rate = param_group['lr']

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        seed_everything(SEED+epoch)
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        for tr_batch_i, (X, truth_label, truth_mask, infor) in enumerate(train_dataloader):
            
            if (lr_scheduler_each_iter):
                scheduler.step(tr_batch_i)

            model.train() 

            X = X.cuda().float()  
            truth_label = truth_label.cuda()
            truth_mask  = truth_mask.cuda()
            prediction = model(X)  # [N, C, H, W]
            
            loss = SoftDiceLoss_binary()(prediction, truth_mask) + \
                   criterion_mask(prediction, truth_mask, weight=None)

            with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                scaled_loss.backward()

            #loss.backward()
        
            if ((tr_batch_i+1) % accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('train_loss', loss.item(), (epoch-1)*len(train_dataloader)*batch_size+tr_batch_i*batch_size)
            
            # print statistics  --------
            probability_mask  = torch.sigmoid(prediction)
            probability_label = probability_mask_to_label(probability_mask)
            # tn, tp, num_neg, num_pos = metric_label(probability_label, truth_label)
            dn, dp, num_neg, num_pos = metric_mask(probability_mask, truth_mask)
            
            l = np.array([ loss.item() * batch_size, dn.sum(), *dp ])
            n = np.array([ batch_size, num_neg.sum(), *num_pos ])
            sum_train_loss += l
            sum_train      += n
            
            # log for training
            if (tr_batch_i+1) % log_step == 0:  
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0
                log.write('lr: %f train loss: %f dn: %f dp1: %f dp2: %f dp3: %f dp4: %f\n' % \
                    (rate, train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5]))
            

            if (tr_batch_i+1) % eval_step == 0: 
                
                eval_count += 1 
                
                valid_loss = np.zeros(17, np.float32)
                valid_num  = np.zeros_like(valid_loss)
                valid_metric = []
                
                with torch.no_grad():
                    
                    torch.cuda.empty_cache()

                    for val_batch_i, (X, truth_label, truth_mask, infor) in enumerate(valid_dataloader):

                        model.eval()

                        X = X.cuda().float()  
                        truth_label = truth_label.cuda()
                        truth_mask  = truth_mask.cuda()
                        prediction = model(X)  # [N, C, H, W]

                        loss = SoftDiceLoss_binary()(prediction, truth_mask) + \
                            criterion_mask(prediction, truth_mask, weight=None)
                            
                        writer.add_scalar('val_loss', loss.item(), (eval_count-1)*len(valid_dataloader)*valid_batch_size+val_batch_i*valid_batch_size)
                        
                        # print statistics  --------
                        probability_mask  = torch.sigmoid(prediction)
                        probability_label = probability_mask_to_label(probability_mask)
                        tn, tp, _, _ = metric_label(probability_label, truth_label)
                        dn, dp, num_neg, num_pos = metric_mask(probability_mask, truth_mask)

                        #---
                        l = np.array([ loss.item()*valid_batch_size, *tn, *tp, *dn, *dp])
                        n = np.array([ valid_batch_size, *num_neg, *num_pos, *num_neg, *num_pos])
                        valid_loss += l
                        valid_num  += n
                        
                    valid_loss = valid_loss / valid_num
                    
                    #------
                    test_pos_ratio = np.array(
                        [NUM_TEST_POS[c][0] / NUM_TEST for c in list(CLASSNAME_TO_CLASSNO.keys())]
                    )
                    test_neg_ratio = 1-test_pos_ratio

                    tn, tp, dn, dp = valid_loss[1:].reshape(-1, NUM_CLASS)
                    kaggle = test_neg_ratio*tn + test_neg_ratio*(1-tn)*dn + test_pos_ratio*tp*dp
                    kaggle = kaggle.mean()

                    kaggle1 = test_neg_ratio*tn + test_pos_ratio*tp
                    kaggle1 = kaggle1.mean()
                    
                    log.write('kaggle value: %f validation loss: %f tn1: %f tn2: %f tn3: %f tn4: %f tp1: %f tp2: %f tp3: %f tp4: %f dn1: %f dn2: %f dn3: %f dn4: %f dp1: %f dp2: %f dp3: %f dp4: %f\n' % \
                    (kaggle1, valid_loss[0], \
                    valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], \
                    valid_loss[5], valid_loss[6], valid_loss[7], valid_loss[8], \
                    valid_loss[9], valid_loss[10], valid_loss[11], valid_loss[12], \
                    valid_loss[13], valid_loss[14], valid_loss[15], valid_loss[16]))


        val_metric_epoch = valid_loss[0]

        if (val_metric_epoch <= valid_metric_optimal):
            
            log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)
            
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    unet_training(args.model, args.model_type, args.optimizer, args.lr_scheduler, args.lr, args.batch_size, args.valid_batch_size, \
                    args.num_epoch, args.start_epoch, args.accumulation_steps, args.train_data_folder, \
                    args.checkpoint_folder, args.load_pretrain)
