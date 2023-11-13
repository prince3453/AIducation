import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define the CNN Model
class FacialExpressionCNN(nn.Module):
    def __init__(self):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Change from 1 to 3 for RGB images
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adjust the linear layer if the pooling layers change
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

    dataset = datasets.ImageFolder(root='Dataset', transform=data_transform)
    class_index_mapping = get_class_index_mapping('Dataset')
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
epochs = 100
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
torch.save(model.state_dict(), 'facial_expression_model.pth')




