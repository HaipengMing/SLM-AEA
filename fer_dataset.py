import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FerDatasetTrain(Dataset):
    def __init__(self, pickled_data, num_classes, k, p, transform=None):
        self.transform = transform
        self.aug_func = add_gaussian_noise
        self.images = pickled_data['images']
        self.labels = torch.tensor(pickled_data['labels'])
        self.labels_one = torch.stack([one_hot(lb, num_classes) for lb in self.labels])
        self.labels_mid = self.labels_one * torch.tensor(k)
        self.labels_mid = ls_p(self.labels_mid, p)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        labels_mid = self.labels_mid[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.aug_func(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, labels_mid, idx


class FerDatasetVal(Dataset):
    def __init__(self, pickled_data, transform=None):
        self.transform = transform
        self.aug_func = add_gaussian_noise
        self.images = pickled_data['images']
        self.labels = torch.tensor(pickled_data['labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def one_hot(x, c):
    res = torch.zeros(c)
    res[x] = 1
    return res


def add_gaussian_noise(image, mean=0.0, var=30, p=0.5):
    if random.uniform(0, 1) < p:
        image_array = np.array(image)
        std = var ** 0.5
        noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img_clipped).convert('RGB')
        return noisy_img
    else:
        return image


def ls_p(labels_mid, p):
    k = torch.max(labels_mid)
    num_classes = labels_mid.size(1)
    c = np.exp(k) * (1 - p) / p
    for _mid in labels_mid:
        _, label = torch.max(_mid, dim=0)
        label = label.item()
        for i in range(num_classes):
            if i == label:
                continue
            else:
                _mid[i] = np.log(c / (num_classes - 1))
    return labels_mid
