import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


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

# Function to load the model
def load_model(model_path):
    model = FacialExpressionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),  # Adjust if your model expects a different size
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to make a prediction
def predict(image_path, model_path):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Example usage
path = 'Dataset/validation/bored/22.jpg'
predicted_class = predict(path, 'variant2.pth')
thisdict = {
  "0": "Angry",
  "1": "Bored",
  "2": "Focused",
   "3" :"neutral"
}
print("Predicted class:", thisdict.get(str(predicted_class)))
