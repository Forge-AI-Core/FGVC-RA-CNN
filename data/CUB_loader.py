"""
CUB-200-2011 Dataset Loader using Hugging Face `datasets`.

Note: 
- This loader requires the dataset to be in Hugging Face Arrow format.
- The default dataset path is expected to be `./data/cub-200-2011`.
- For other users, they only need to run `uv sync` to install all dependencies and then the dataset can be loaded instantly.
"""
# crop size 448
# mean 109.973, 127.336, 123.883
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from datasets import load_from_disk

class CUB200_loader(data.Dataset):
    def __init__(self, root, split = 'train', transform = None):
        std = 1. / 255.
        means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]

        self.dataset = load_from_disk(root)[split]
        
        if transform is None and split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(448),
                transforms.RandomCrop([448, 448]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = means,
                    std = [std]*3)
                ])
        elif transform is None and split.lower() == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(448),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = means,
                    std = [std]*3)
                ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item['image'].convert('RGB')
        img = self.transform(img)
        cls = item['label']
        return img, cls

    def __len__(self):
        return len(self.dataset)

    def CUB_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls

if __name__ == '__main__':
    # Usage example
    trainset = CUB200_loader('./cub-200-2011', split='train')
    trainloader = data.DataLoader(trainset, batch_size=32,
            shuffle=False, collate_fn=trainset.CUB_collate, num_workers=1)
    
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        break
