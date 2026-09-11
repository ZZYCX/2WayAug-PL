import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

def read_jpg(img_path):
    return Image.open(img_path).convert('RGB')

class MLCPLDataset(Dataset):
    def __init__(self, dataset_path, records, num_categories, transform, categories=None, read_func=read_jpg):
        self.dataset_path = dataset_path
        self.records = records
        self.categories = categories
        self.num_categories = num_categories
        self.transform = transform
        self.read_func = read_func

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        id, path, pos_category_nos, neg_category_nos, unc_category_nos = self.records[idx]
        img_path = os.path.join(self.dataset_path, path)
        img = self.read_func(img_path)
        img = self.transform(img)
        target = labels_to_one_hot(pos_category_nos, neg_category_nos, unc_category_nos, self.num_categories)
        return img, target
    
    def drop_labels_random(self, target_partial_ratio, seed=526):
        # Evaluation only: never returned by __getitem__ or used as training targets.
        if not hasattr(self, 'ground_truth'):
            self.ground_truth = torch.full((len(self), self.num_categories), -1, dtype=torch.int8)
            for i, (_, _, positives, negatives, _) in enumerate(self.records):
                self.ground_truth[i, positives] = 1
                self.ground_truth[i, negatives] = 0
        self.records = drop_labels(self.records, target_partial_ratio, seed=seed)
        return self
    
def labels_to_one_hot(positives, negatives, uncertains, num_categories):
    one_hot = torch.full((num_categories, ), torch.nan, dtype=torch.float32)
    one_hot[np.array(positives)] = 1.0
    one_hot[np.array(negatives)] = 0.0
    one_hot[np.array(uncertains)] = -1.0
    return one_hot
    
def fill_nan_to_negative(old_records, num_categories):
    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_neg_category_nos = [x for x in range(num_categories) if x not in pos_category_nos+unc_category_nos]
        new_records.append((i, path, pos_category_nos, new_neg_category_nos, unc_category_nos))
    return new_records

def drop_labels(old_records, target_partial_ratio, seed=526):
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_pos_category_nos = [no for no in pos_category_nos if rng.random() < target_partial_ratio]
        new_neg_category_nos = [no for no in neg_category_nos if rng.random() < target_partial_ratio]
        new_unc_category_nos = [no for no in unc_category_nos if rng.random() < target_partial_ratio]
        new_records.append((i, path, new_pos_category_nos, new_neg_category_nos, new_unc_category_nos))
    return new_records