import pandas as pd
import numpy as np
from ml_stratifiers import MultilabelStratifiedKFold


PATH = '/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization'

def get_label(attribute_ids):
    attribute_ids = attribute_ids.split()
    for _,ids in enumerate(attribute_ids):
        attribute_ids[_] = int(ids)
    return attribute_ids


def split(n_splits=5, seed=42):

    image_class_df = pd.read_csv("/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization/image_class.csv")
    
    image_class_df['class'] = image_class_df['class'].apply(get_label)

    labels_encoded = list(image_class_df['class'].values)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed)
    splits = mskf.split(image_class_df['id'], labels_encoded)

    for fold_, (tr, val) in enumerate(splits):
        
        image_train = []
        image_val = []
        
        for i in list(image_class_df.iloc[tr]['id'].values):
            image_train.append([i, 'train'])
        for i in list(image_class_df.iloc[val]['id'].values):
            image_val.append([i, 'train'])
        
        np.save(PATH + '/split/train_fold_%s_seed_%s.npy'%(fold_, seed), image_train)
        np.save(PATH + '/split/val_fold_%s_seed_%s.npy'%(fold_, seed), image_val)
        
        
if __name__ == "__main__":
    split()