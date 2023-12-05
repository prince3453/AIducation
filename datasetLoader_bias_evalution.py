from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
#
def DatasetLoader(root_folder='FinalDataset', batch_size=32, train_size=0.7, val_size=0.15, test_size=0.15):
    DataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=root_folder, transform=DataTransform)

    TotalSize = len(dataset)
    train_size = int(train_size * TotalSize)
    val_size = int(val_size * TotalSize)
    test_size = TotalSize - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
