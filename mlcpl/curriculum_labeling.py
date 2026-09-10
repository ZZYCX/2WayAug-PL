from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from datetime import datetime

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class CurriculumLabeling(Dataset):
    def __init__(self, dataset, transform_for_update=None):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)

        self.transform_for_update = transform_for_update

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], target)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, batch_size=32, num_workers=20, thresholds=(-4, 4), device='cuda', verbose=False):
        temp = self.dataset.transform
        self.dataset.transform = self.transform_for_update

        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        model.eval()

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                if not verbose:
                    print(f'[{timestamp()}] Updating labels batch={batch+1}/{len(dataloader)}')

                x, y = x.to(device), y.to(device)
                logit = model(x)
                
                label = torch.sign(logit)
                label = torch.where(label==-1, 0, label)
                self.labels[batch*batch_size: (batch+1)*batch_size] = label.cpu()

                negative_threshold, positive_threshold = thresholds
                negative_selection = torch.where(logit<negative_threshold, 1, 0)
                positive_selection = torch.where(logit>positive_threshold, 1, 0)

                selection = torch.logical_or(negative_selection, positive_selection)
                
                self.selections[batch*batch_size: (batch+1)*batch_size] = torch.logical_and(selection, torch.isnan(y)).cpu()
        
        if not verbose:
            print()

        self.dataset.transform = temp

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)
