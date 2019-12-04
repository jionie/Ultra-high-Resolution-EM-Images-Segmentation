import numpy as np
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_path', type=str, default='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/train/', \
    required=False, help='specify the data path of images')
parser.add_argument('--label_path', type=str, default='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/labels/train/', \
    required=False, help='specify the label path')
parser.add_argument('--save_path', type=str, default='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE/', \
    required=False, help='specify the train infor csv path')


def get_dataset_info(data_path, label_path, save_path):
    data_list = list(map(lambda x: x.replace('.png', ''), os.listdir(data_path)))
    train = pd.DataFrame(data_list, columns=['id'])
    
    train.to_csv(save_path + 'train.csv', index=False)
    
    return


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    get_dataset_info(args.data_path, args.label_path, args.save_path)

