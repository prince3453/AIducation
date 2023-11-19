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

        # Combine the first two convolution layers into one
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the input size for the first linear layer
        dummy_input = torch.randn(1, 1, 48, 48)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))

        self.fc_input_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def get_class_index_mapping(path):
    dataset = datasets.ImageFolder(root=path)
    return dataset.class_to_idx

# Function to load the dataset
def dataset_loader():
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root='FinalDataset', transform=data_transform)
    class_index_mapping = get_class_index_mapping('FinalDataset')
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = dt.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

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
        # print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')
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
torch.save(model.state_dict(), 'variant1.pth')