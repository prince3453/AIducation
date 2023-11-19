import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class FacialExpressionCNN(nn.Module):
    def __init__(self):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, padding=3)  # Change kernel size to 7x7
        self.conv2 = nn.Conv2d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_class_index_mapping(path):
    dataset = datasets.ImageFolder(root=path)
    return dataset.class_to_idx

#it is the function to load the dataset
def dataset_loader():
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root='FinalDataset', transform=data_transform) #define the directory
    class_index_mapping = get_class_index_mapping('FinalDataset')
    ValidationSize = int(0.2 * len(dataset))
    TrainSize = len(dataset) - ValidationSize

    TrainDataset, ValidationDataset = dt.random_split(dataset, [TrainSize, ValidationSize])

    train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ValidationDataset, batch_size=32, shuffle=True)

    return train_loader, val_loader

# Load datasets
train_loader, val_loader = dataset_loader()

# Initialize the model
model = FacialExpressionCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1} loss: {running_loss / len(train_loader)}')

# Model evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the validation set: {100 * correct / total}%')
# Assuming 'model' is your trained model instance
torch.save(model.state_dict(), 'variant2.pth')