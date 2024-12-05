import torch
import os
from PIL import Image
from torchvision.transforms import functional as F
import train_scripts.transforms as T


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, split="train"):
        self.root = root
        self.transforms = transforms
        self.split = split

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, f"{split}/images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, f"{split}/labels"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, f"{self.split}/images", self.imgs[idx])
        bxs_path = os.path.join(self.root, f"{self.split}/labels", self.masks[idx])
        img = Image.open(img_path).convert("RGB").resize((400,400))

        with open(bxs_path) as f:
            data = f.readlines()

        # get bounding box coordinates for each mask
        num_objs = len(data)
        boxes = []
        for i in range(num_objs):
            values = data[i].split()
            xc = float(values[1])
            yc = float(values[2])
            w = float(values[3])
            h = float(values[4])
            xmin = (xc - w/2)*400
            xmax = (xc + w/2)*400
            ymin = (yc - h/2)*400
            ymax = (yc + h/2)*400
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




def get_datasets(name):
    dataset_train = PennFudanDataset(f'datasets/{name}', get_transform(train=True))
    dataset_test = PennFudanDataset(f'datasets/{name}', get_transform(train=False), split="valid")

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices)
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices)
    return dataset_train, dataset_test


def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loader(dataset, batch, shuffle):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn)
    return data_loader