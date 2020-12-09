from torch.utils.data import Dataset
import random
import pandas as pd
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class SeverstalSteel(Dataset):
    def __init__(self, root_dir, split, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w
        self.split = split
        if split in ["train", "val"]:
            self.labels = pd.read_csv(os.path.join(root_dir, "train.csv"), index_col="ImageId")
            self.image_files = sorted(glob(os.path.join(root_dir, "train_images", "*.jpg")))
            if split == "train":
                self.image_files = random.choices(self.image_files, k=int(0.8*len(self.image_files)))
            else:
                self.image_files = random.choices(self.image_files, k=int(0.2*len(self.image_files)))
        elif split == "test":
            self.labels = None
            self.image_files = sorted(glob(os.path.join(root_dir, "test_images", "*.jpg")))
        else:
            raise ValueError("invalid split")
        # for now!
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=Image.NEAREST),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3443, 0.3443, 0.3443], std=[0.1507, 0.1507, 0.1507]),
        ])
    
    def __getitem__(self, indx):
        image_file = self.image_files[indx]
        img = Image.open(image_file).convert('RGB')
        wt, ht = img.size
        img = self.img_transform(img)
        
        mask = np.zeros((ht, wt))
        if self.labels is not None:
            if os.path.basename(image_file) in self.labels.index.tolist():
                label = self.labels.loc[[os.path.basename(image_file)]]
                class_ind = label["ClassId"].tolist()
                encoded_pix = label["EncodedPixels"].tolist()
                for i in range(len(class_ind)):
                    e = encoded_pix[i].split()
                    pos, length = map(int, e[0::2]), map(int, e[1::2])
                    temp = np.zeros(ht*wt)
                    for (p,l) in zip(pos, length):
                        temp[p:p+l] = class_ind[i]
                    mask += temp.reshape(ht, wt, order="F")
            mask = self.mask_transform(torch.from_numpy(mask).unsqueeze(0))
            mask = torch.squeeze(mask)
            if random.random() < 0.4 and self.split == "train":
                img = torch.flip(img, [-1])
                mask = torch.flip(mask, [-1])
            return img, mask
        else:
            return img

    def __len__(self):
        return len(self.image_files)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm
    dataset = SeverstalSteel("severstal-steel-defect-detection", "train", 64, 512)
    dataloader = DataLoader(dataset, batch_size=32)
    # n_images = 0
    # mean = 0
    # var = 0
    # pbar = tqdm(total=len(dataloader))
    # for (img, _) in dataloader:
    #     img = img.view(img.shape[0], img.shape[1], -1)
    #     n_images += img.size(0)
    #     mean += img.mean(2).sum(0) 
    #     var += img.var(2).sum(0)
    #     pbar.update(1)
    # pbar.close()
    # mean /= n_images
    # var /= n_images
    # std = torch.sqrt(var)
    # print(mean)
    # print(std)
    # out = dataset[0]
    # pbar = tqdm(total=len(dataloader))
    # class_counts = {c:0 for c in range(5)}
    # for (mask) in dataloader:
    #     ind, count = np.unique(mask.numpy(), return_counts=True)
    #     for i in range(len(ind)):
    #         class_counts[ind[i]] += count[i]
    #     pbar.update(1)
    # print(class_counts)
    # c = list(class_counts.values())
    # print([i/sum(c) for i in c])
