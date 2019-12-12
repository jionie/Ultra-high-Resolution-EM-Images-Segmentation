# import os and define graphic card
import os
os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from functools import partial

# import pytorch related libraries
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# import apex for mix precision training
from apex import amp

# import albumentations for augmentation
import albumentations
from albumentations import torch as AT

# import split dataset
from generating_dataset.split_data import split

# import dataset class
from dataset.dataset import URESDataset
from dataset.dataset import null_collate

# import utils
from utils.transform import *
from utils.ranger import *
import utils.learning_schedules_fastai as lsf
from utils.fastai_optim import OptimWrapper
from utils.lrs_scheduler import * 
from utils.other_loss import *
from utils.lovasz_loss import *
from utils.loss_function import *
from utils.metric import *

from utils.file import *

# import aspp model related libraries
from segmentation.aspp.model import *

# import deeplab model related libraries
from segmentation.deeplab_v3.semantic_segmentation.network.deepv3 import *
from segmentation.deeplab_v3.ef_unet import *

# import unet model related libraries
from segmentation.unet.models.model import *


############################################################################## define augument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model', type=str, default='efficientnet-b5', required=False, help='specify the backbone model')
parser.add_argument('--model_type', type=str, default='unet', required=False, help='specify the model')
parser.add_argument('--optimizer', type=str, default='Ranger', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmRestart', required=False, help="specify the lr scheduler")
parser.add_argument("--lr", type=int, default=2e-3, required=False, help="specify the initial learning rate for training")
parser.add_argument("--batch_size", type=int, default=2, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=1, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=15, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=2, required=False, help="specify the accumulation steps")
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--train_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE", \
    required=False, help="specify the folder for training data")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/URES/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')


############################################################################## define constant
SEED = 42
N_SPLITS = 5
NUM_CLASS = 1
MASK_WIDTH = 1024
MASK_HEIGHT = 1024
DATA_DIR = '/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE'



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
def transform_train(image, mask, infor):

    if random.random() < 0.5:
        image = albumentations.VerticalFlip(p=1)(image=image)['image']
        mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        
        image = albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.1),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.1),
        ])(image=image)['image']
    
    if random.random() < 0.5:  
        image = albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5)(image=image)['image']

    if random.random() < 0.5:
        image = albumentations.Cutout(num_holes=2, max_h_size=8, max_w_size=8, p=1)(image=image)['image']
        mask = albumentations.Cutout(num_holes=2, max_h_size=8, max_w_size=8, p=1)(image=mask)['image']
        
    if random.random() < 0.5:
        image = albumentations.RandomRotate90(p=1)(image=image)['image']
        mask = albumentations.RandomRotate90(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.Transpose(p=1)(image=image)['image']
        mask = albumentations.Transpose(p=1)(image=mask)['image']
    
    if random.random() < 0.5:
        image = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=image)['image']
        mask = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=mask)['image']
    
    # no normalization for now
    # image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']
    image = albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0)(image=image)['image']
    
    return image, mask, infor

def transform_valid(image, mask, infor):
    
    # no normalization for now
    # image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']
    image = albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0)(image=image)['image']

    return image, mask, infor


############################################################################## define function for fast.ai adam
def children(m: nn.Module):
    return list(m.children())

def num_children(m: nn.Module):
    return len(children(m))



############################################################################## define function for training
def training(model_name,
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
            train_split,
            val_split,
            fold,
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
    
    os.makedirs(checkpoint_folder + '/' + model_type + '/' + model_name, exist_ok=True)
    
    log = Logger()
    log.open(checkpoint_folder + '/' + model_type + '/' + model_name + '/' + model_name + '_fold_' + str(fold) + '_log_train.txt', mode='a+')
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % train_data_folder)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % checkpoint_folder)
    log.write('\n')

    
    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = URESDataset(
        data_dir = train_data_folder,
        mode     = 'train',
        csv      = ['train.csv',],
        split    = train_split,
        augment  = transform_train,
        size=(1024, 1024),
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

    valid_dataset = URESDataset(
        data_dir = train_data_folder,
        mode    = 'train',
        csv     = ['train.csv',],
        split   = val_split,
        augment = transform_valid,
        size=(1024, 1024),
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
    def load(model, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)
        model.load_state_dict(state_dict)
        
        return model
    
    def get_deeplab_model(model_name="deep_se101", in_channel=3, num_classes=1, criterion=SoftDiceLoss_binary(), \
            load_pretrain=False, checkpoint_filepath=None):
        
        if model_name == 'deep_se50':
            model = DeepSRNX50V3PlusD_m1(in_channel=in_channel, num_classes=num_classes, criterion=criterion)
        elif model_name == 'deep_se101':
            model = DeepSRNX101V3PlusD_m1(in_channel=in_channel, num_classes=num_classes, criterion=criterion)
        elif model_name == 'WideResnet38':
            model = DeepWR38V3PlusD_m1(in_channel=in_channel, num_classes=num_classes, criterion=criterion)
        elif model_name == 'unet_ef3':
            model = EfficientNet_3_unet()
        elif model_name == 'unet_ef5':
            model = EfficientNet_5_unet()
        else:
            print('No model name in it')
            model = None
            
        if (load_pretrain):
            model = load(model, checkpoint_filepath)
        
        return model
    
    def get_unet_model(model_name="efficientnet-b3", IN_CHANNEL=3, NUM_CLASSES=1, \
            WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT, load_pretrain=False, checkpoint_filepath=None):
        
        model = model_iMet(model_name, IN_CHANNEL, NUM_CLASSES, WIDTH, HEIGHT)
        
        if (load_pretrain):
            model.load_pretrain(checkpoint_filepath)
        
        return model
    
    def get_aspp_model(model_name="efficientnet-b3", NUM_CLASSES=1, load_pretrain=False, checkpoint_filepath=None):
        
        model = Net(model_name, NUM_CLASSES)
        if (load_pretrain):
            state_dict = torch.load(checkpoint_filepath, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=True)
        
        return model

    ############################################################################### training parameters
    checkpoint_filename = model_type + '/' + model_name + '/' + model_name + "_" + model_type + '_fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################### model and optimizer
    if model_type == 'unet': 
        model = get_unet_model(model_name=model_name, IN_CHANNEL=3, NUM_CLASSES=NUM_CLASS, \
            WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT, load_pretrain=load_pretrain, checkpoint_filepath=checkpoint_filepath)
    elif model_type == 'deeplab':
        model = get_deeplab_model(model_name=model_name, in_channel=3, num_classes=NUM_CLASS, \
            criterion=BCEDiceLoss(), load_pretrain=load_pretrain, checkpoint_filepath=checkpoint_filepath)
    elif model_type == 'aspp':
        model = get_aspp_model(model_name=model_name, NUM_CLASSES=NUM_CLASS, \
            load_pretrain=load_pretrain, checkpoint_filepath=checkpoint_filepath)
    
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
    
    # define criterion
    criterion = BCEDiceLoss()
    metric = FscoreMetric()
    
    for epoch in range(1, num_epoch+1):
        
        # update lr and start from start_epoch  
        if (not lr_scheduler_each_iter):
            if epoch < 6:
                if epoch != 0:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2) 
            elif epoch > 5 and epoch < 8:
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
        
        for tr_batch_i, (X, truth_mask, infor) in enumerate(train_dataloader):
            
            if (lr_scheduler_each_iter):
                scheduler.step(tr_batch_i)

            model.train() 

            X = X.cuda().float()  
            truth_mask  = truth_mask.cuda()
            prediction = model(X)  # [N, C, H, W]
            loss = criterion(prediction, truth_mask, weight=None)

            with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                scaled_loss.backward()

            #loss.backward()
        
            if ((tr_batch_i+1) % accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('train_loss_' + str(fold), loss.item(), (epoch-1)*len(train_dataloader)*batch_size+tr_batch_i*batch_size)
            
            # print statistics  --------
            probability_mask  = torch.sigmoid(prediction)
            fscore_positive = metric(probability_mask, truth_mask)
            fscore_negative = metric(probability_mask, torch.ones_like(truth_mask) - truth_mask)
            
            l = np.array([loss.item() * batch_size, fscore_positive, fscore_negative])
            n = np.array([batch_size])
            sum_train_loss += l
            sum_train      += n
            
            # log for training
            if (tr_batch_i+1) % log_step == 0:  
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0
                log.write('lr: %f train loss: %f fscore_positive: %f fscore_negative: %f\n' % \
                    (rate, train_loss[0], train_loss[1], train_loss[2]))
            

            if (tr_batch_i+1) % eval_step == 0:  
                
                eval_count += 1
                
                valid_loss = np.zeros(17, np.float32)
                valid_num  = np.zeros_like(valid_loss)
                valid_metric = []
                
                with torch.no_grad():
                    
                    torch.cuda.empty_cache()

                    for val_batch_i, (X, truth_mask, infor) in enumerate(valid_dataloader):

                        model.eval()

                        X = X.cuda().float()  
                        truth_mask  = truth_mask.cuda()
                        prediction = model(X)  # [N, C, H, W]

                        #SoftDiceLoss_binary()(prediction, truth_mask)
                        loss = criterion(prediction, truth_mask, weight=None)
                            
                        writer.add_scalar('val_loss_' + str(fold), loss.item(), (eval_count-1)*len(valid_dataloader)*valid_batch_size+val_batch_i*valid_batch_size)
                        
                        # print statistics  --------
                        probability_mask  = torch.sigmoid(prediction)
                        fscore_positive = metric(probability_mask, truth_mask)
                        fscore_negative = metric(probability_mask, torch.ones_like(truth_mask) - truth_mask)

                        #---
                        l = np.array([loss.item()*valid_batch_size, fscore_positive, fscore_negative])
                        n = np.array([valid_batch_size])
                        valid_loss += l
                        valid_num  += n
                        
                    valid_loss = valid_loss / valid_num
                    
                    log.write('validation loss: %f fscore_positive: %f fscore_negative: %f\n' % \
                    (valid_loss[0], \
                    valid_loss[1], \
                    valid_loss[2]))


        val_metric_epoch = valid_loss[0]

        if (val_metric_epoch <= valid_metric_optimal):
            
            log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    split(train_info_path='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/train.csv', \
        save_path='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/', \
        n_splits=N_SPLITS, \
        seed=SEED)
    
    for fold_ in range(N_SPLITS):
        
        train_split = ['train_fold_%s_seed_%s.npy'%(fold_, SEED)]
        val_split = ['val_fold_%s_seed_%s.npy'%(fold_, SEED)]
    
        training(args.model, args.model_type, args.optimizer, args.lr_scheduler, args.lr, args.batch_size, args.valid_batch_size, \
                        args.num_epoch, args.start_epoch, args.accumulation_steps, args.train_data_folder, \
                        args.checkpoint_folder, train_split, val_split, fold_, args.load_pretrain)