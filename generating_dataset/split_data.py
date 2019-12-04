import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from ml_stratifiers import MultilabelStratifiedKFold


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--train_info_path', type=str, default='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/', \
    required=False, help='specify the path for train.csv')
parser.add_argument('--n_splits', type=int, default=5, \
    required=False, help='specify the number of folds')
parser.add_argument('--seed', type=int, default=42, \
    required=False, help='specify the random seed for splitting dataset')
parser.add_argument('--save_path', type=str, default='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/', \
    required=False, help='specify the path for train.csv')

def get_label(attribute_ids):
    attribute_ids = attribute_ids.split()
    for _,ids in enumerate(attribute_ids):
        attribute_ids[_] = int(ids)
    return attribute_ids


def split(train_info_path, save_path, n_splits=5, seed=42):

    os.makedirs(save_path + '/split', exist_ok=True)
    image_class_df = pd.read_csv(train_info_path)

    kf = KFold(n_splits=n_splits, random_state=seed)
    splits = kf.split(image_class_df['id'])

    for fold_, (tr, val) in enumerate(splits):
        
        image_train = []
        image_val = []
        
        for i in list(image_class_df.iloc[tr]['id'].values):
            image_train.append([i, 'train'])
        for i in list(image_class_df.iloc[val]['id'].values):
            image_val.append([i, 'train'])
        
        np.save(save_path + '/split/train_fold_%s_seed_%s.npy'%(fold_, seed), image_train)
        np.save(save_path + '/split/val_fold_%s_seed_%s.npy'%(fold_, seed), image_val)
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    split(args.train_info_path, args.save_path, args.n_splits, args.save_path)