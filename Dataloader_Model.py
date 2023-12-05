import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def DatasetLoader(train_folder, val_folder, batch_size=32):
    DataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    TrainDataset = datasets.ImageFolder(root=train_folder, transform=DataTransform)
    ValDataset = datasets.ImageFolder(root=val_folder, transform=DataTransform)

    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=True)


    return TrainLoader, ValLoader

