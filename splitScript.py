import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def SplitDataset(root_folder, output_folder, train_ratio=0.7, val_ratio=0.15):
    # Define transform
    DataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Create output directories
    TrainDir = os.path.join(output_folder, 'train')
    ValDir = os.path.join(output_folder, 'validation')
    TestDir = os.path.join(output_folder, 'test')

    for DirPath in [TrainDir, ValDir, TestDir]:
        for EmotionFolder in ['angry', 'bored', 'focused', 'neutral']:
            os.makedirs(os.path.join(DirPath, EmotionFolder), exist_ok=True)

    # Split the dataset
    for EmotionFolder in ['angry', 'bored', 'focused', 'neutral']:
        EmotionPath = os.path.join(root_folder, EmotionFolder)
        files = os.listdir(EmotionPath)
        NumFiles = len(files)

        TrainSplit = int(train_ratio * NumFiles)
        ValSplit = int((train_ratio + val_ratio) * NumFiles)

        TrainFiles = files[:TrainSplit]
        ValFiles = files[TrainSplit:ValSplit]
        TestFiles = files[ValSplit:]

        # Copy files to the respective directories
        for file in TrainFiles:
            SrcPath = os.path.join(EmotionPath, file)
            DestPath = os.path.join(TrainDir, EmotionFolder, file)
            shutil.copy(SrcPath, DestPath)

        for file in ValFiles:
            SrcPath = os.path.join(EmotionPath, file)
            DestPath = os.path.join(ValDir, EmotionFolder, file)
            shutil.copy(SrcPath, DestPath)

        for file in TestFiles:
            SrcPath = os.path.join(EmotionPath, file)
            DestPath = os.path.join(TestDir, EmotionFolder, file)
            shutil.copy(SrcPath, DestPath)

    # Create and return DataLoader
    TrainDataset = datasets.ImageFolder(root=TrainDir, transform=DataTransform)
    ValDataset = datasets.ImageFolder(root=ValDir, transform=DataTransform)
    TestDataset = datasets.ImageFolder(root=TestDir, transform=DataTransform)

    TrainLoader = DataLoader(TrainDataset, batch_size=32, shuffle=True)
    ValLoader = DataLoader(ValDataset, batch_size=32, shuffle=True)
    TestLoader = DataLoader(TestDataset, batch_size=32, shuffle=True)

    return TrainLoader, ValLoader, TestLoader

