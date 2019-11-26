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

from models.model import *


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


############################################################################## define augument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model_name', type=str, default='efficientnet-b5', required=False, help='specify the backbone model')
parser.add_argument('--model_type', type=str, default='unet', required=False, help='specify the segmentation model, deeplab, unet or fpn')
parser.add_argument('--mode', type=str, default='test', required=False, help='specify the mode, valid or test')
parser.add_argument("--is_save", action='store_true', default=True, help="whether to save predicted result as npy")
parser.add_argument("--batch_size", type=int, default=32, required=False, help="specify the batch size for dataloader")
parser.add_argument("--test_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization", \
    required=False, help="specify the folder for data")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Cloud/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument("--result_folder", type=str, default="/media/jionie/my_disk/Kaggle/Cloud/result", \
    required=False, help="specify the folder for result")
parser.add_argument("--train_split", type=list, default=['by_random1/valid_fold_a0_300.npy',], \
    required=False, help="specify the validation set for validation")
parser.add_argument("--augment", type=list, default=['null', 'flip_lr', 'flip_ud', 'flip_both'], \
    required=False, help="specify the augment for tta")



############################################################################## define transform for normalize
def transform_valid(image, label, mask, infor):
        
    image = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']

    return image, label, mask, infor

############################################################################## define evaluation function
def do_evaluate_segmentation(net, test_dataset, batchsize, augment=[], out_dir=None):
    
    test_loader = DataLoader(
        test_dataset,
        sampler     = SequentialSampler(test_dataset),
        batch_size  = batchsize,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    #----



    test_num  = 0
    test_id   = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = [] # 8bit
    test_truth_mask  = [] # 8bit

    for batch_i, (input, truth_label, truth_mask, infor) in enumerate(test_loader):
        
        if (batch_i % 10 == 0):
            print("processing", batch_i, "batch of", len(test_loader), "batches")

        batch_size, C , H, W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  net(input)
                probability = torch.sigmoid(logit)

                probability_mask  = probability
                probability_label = probability_mask_to_label(probability)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = net(torch.flip(input, dims=[3]))
                probability = torch.sigmoid(torch.flip(logit, dims=[3]))

                probability_mask  += probability
                probability_label += probability_mask_to_label(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = net(torch.flip(input, dims=[2]))
                probability = torch.sigmoid(torch.flip(logit, dims=[2]))

                probability_mask += probability
                probability_label+= probability_mask_to_label(probability)
                num_augment+=1
                
            if 'flip_both' in augment:
                logit = net(torch.flip(input, dims=[2, 3]))
                probability = torch.sigmoid(torch.flip(logit, dims=[2, 3]))

                probability_mask += probability
                probability_label+= probability_mask_to_label(probability)
                num_augment+=1

            #---
            probability_mask  = probability_mask/num_augment
            probability_label = probability_label/num_augment

        #---
        batch_size  = len(infor)
        truth_label = truth_label.data.cpu().numpy().astype(np.uint8)
        truth_mask  = truth_mask.data.cpu().numpy().astype(np.uint8)
        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)
        probability_label = (probability_label.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i.image_id for i in infor])
        test_truth_label.append(truth_label)
        test_truth_mask.append(truth_mask)
        test_probability_label.append(probability_label)
        test_probability_mask.append(probability_mask)
        test_num += batch_size


    assert(test_num == len(test_loader.dataset))

    test_truth_label = np.concatenate(test_truth_label)
    test_truth_mask  = np.concatenate(test_truth_mask)
    test_probability_label = np.concatenate(test_probability_label)
    test_probability_mask = np.concatenate(test_probability_mask)

    return test_id, test_truth_label, test_truth_mask, test_probability_label, test_probability_mask


######################################################################################
def run_submit_segmentation(model_name,
                            model_type,
                            mode,
                            is_save,
                            batch_size,
                            test_data_folder, 
                            result_folder,
                            checkpoint_folder,
                            train_split=['by_random1/valid_fold_a0_300.npy',],
                            augment=['null', 'flip_lr', 'flip_ud']):

    out_dir = \
        result_folder + '/' + model_type + '/' + model_name + '/' + mode + '/'
    initial_checkpoint = \
        checkpoint_folder + '/' + model_type + '/' + model_name + '/' + model_name + '_' + model_type + '_checkpoint.pth'


    ###############################################################
    if len(augment) > 2:
        mode_folder = 'test-tta' #tta
    else:
        mode_folder = 'test' #null

    #---

    ## setup
    os.makedirs(out_dir +'/submit/%s'%(mode_folder), exist_ok=True)
    
    
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

    log = Logger()
    log.open(out_dir+'/' + model_name + '_log_submit.txt',mode='a')
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'valid':
        test_dataset = CloudDataset(
            data_dir = test_data_folder,
            mode    = 'train',
            csv     = ['train.csv',],
            split   = train_split,
            augment = transform_valid,
        )

    if mode == 'test':
        test_dataset = CloudDataset(
            data_dir = test_data_folder,
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_3698.npy',],
            augment = transform_valid, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)


    ## start testing here! ##############################################
    #

    #---
    threshold_label      = [ 0.80, 0.80, 0.80, 0.80,]
    threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
    threshold_mask_size  = [   1,   1,   1,   1,]
    
    ############################################################################## define unet model with backbone
    MASK_WIDTH = 525
    MASK_HEIGHT = 350
    
    def get_unet_model(model_name="efficientnet-b3", IN_CHANNEL=3, NUM_CLASSES=2, WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT):
        model = model_iMet(model_name, IN_CHANNEL, NUM_CLASSES, WIDTH, HEIGHT)
        
        # Optional, for multi GPU training and inference
        # model = nn.DataParallel(model)
        return model


    if is_save: #save
        ## net ----------------------------------------
        log.write('** net setting **\n')

        model = get_unet_model(model_name=model_name, IN_CHANNEL=3, NUM_CLASSES=len(CLASSNAME_TO_CLASSNO), WIDTH=MASK_WIDTH, HEIGHT=MASK_HEIGHT)
        model.load_pretrain(initial_checkpoint)
        model = model.cuda()    

        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        log.write('\n')


        image_id, truth_label, truth_mask, probability_label, probability_mask,  =\
            do_evaluate_segmentation(model, test_dataset, batchsize, augment)

        write_list_to_file (out_dir + '/submit/%s/image_id.txt'%(mode_folder),image_id)
        np.savez_compressed(out_dir + '/submit/%s/probability_label.uint8.npz'%(mode_folder), probability_label)
        np.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode_folder), probability_mask)
        if mode == 'valid':
            np.savez_compressed(out_dir + '/submit/%s/truth_label.uint8.npz'%(mode_folder), truth_label)
            np.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode_folder), truth_mask)

    else:
        image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt'%(mode_folder))
        probability_label = np.load(out_dir + '/submit/%s/probability_label.uint8.npz'%(mode_folder))['arr_0']
        probability_mask  = np.load(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode_folder))['arr_0']
        if mode == 'valid':
            truth_label = np.load(out_dir + '/submit/%s/truth_label.uint8.npz'%(mode_folder))['arr_0']
            truth_mask  = np.load(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode_folder))['arr_0']

    num_test= len(image_id)
    # if 0: #show
    #     if mode == 'train':
    #         folder='train_images'
    #         for b in range(num_test):
    #             print(b, image_id[b])
    #             image=cv2.imread(DATA_DIR+'/%s/%s'%(folder,image_id[b]), cv2.IMREAD_COLOR)
    #             result = draw_predict_result(
    #                 image,
    #                 truth_label[b],
    #                 truth_mask[b],
    #                 probability_label[b].astype(np.float32)/255,
    #                 probability_mask[b].astype(np.float32)/255
    #             )
    #             image_show('result',result,0.5)
    #             cv2.waitKey(0)

    #----
    if 1: #decode

        # value = np.max(probability_mask,1,keepdims=True)
        # value = probability_mask*(value==probability_mask)
        pass



    # inspect here !!!  ###################
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('threshold_label = %s\n'%str(threshold_label))
    log.write('threshold_mask_pixel = %s\n'%str(threshold_mask_pixel))
    log.write('threshold_mask_size  = %s\n'%str(threshold_mask_size))
    log.write('\n')

    if mode == 'valid':

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)


        log.write('** threshold_label **\n')
        result = compute_metric(predict_label, predict_mask, truth_label, truth_mask)
        text = summarise_metric(result)
        log.write('\n%s'%(text))
        
        
        #-----

        # auc, result = compute_roc_label(truth_label, probability_label)
        # text = summarise_roc_label(auc, result)
        # log.write('\n%s'%(text))


        #-----

        log.write('** threshold_pixel + threshold_label **\n')
        predict_mask = predict_mask * predict_label.reshape(-1,4,1,1)
        result = compute_metric(predict_label, predict_mask, truth_label, truth_mask)
        text = summarise_metric(result)
        log.write('\n%s'%(text))

        #-----

        # log.write('** threshold_pixel + threshold_label + threshold_size **\n')
        # predict_mask = remove_small(predict_mask, threshold_mask_size)
        # result = compute_metric(predict_label, predict_mask, truth_label, truth_mask)
        # text = summarise_metric(result)
        # log.write('\n%s'%(text))


    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        file_name = model_type + '_' + model_name + '.csv'
        csv_file = out_dir +'/submit/%s/'%(mode_folder) + file_name

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(NUM_CLASS):
                image_id_class_id.append(image_id[b]+'_%s'%(CLASSNO_TO_CLASSNAME[c]))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['Image_Label', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)

        ## print statistics ----
        print('initial_checkpoint=%s'%initial_checkpoint)
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))

        ##evalue based on probing results
        # text = do_local_submit(image_id, predict_label,predict_mask)
        # log.write('\n')
        # log.write('%s'%(text))


        #--
        local_result = find_local_threshold(image_id, probability_label, cutoff=[90,0,550,110])
        threshold_label = [local_result[0][0],local_result[1][0],local_result[2][0],local_result[3][0]]
        log.write('test threshold_label=%s\n'%str(threshold_label))

        # predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        # text = do_local_submit(image_id, predict_label,predict_mask=None)
        # log.write('\n')
        # log.write('%s'%(text))

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    run_submit_segmentation(args.model_name, \
                            args.model_type,
                            args.mode,
                            args.is_save,
                            args.batch_size,
                            args.test_data_folder, 
                            args.result_folder,
                            args.checkpoint_folder,
                            args.train_split,
                            args.augment)
    
   