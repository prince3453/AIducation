import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


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
path = 'Dataset/train/focused/22.jpg'
predicted_class = predict(path, 'variant1.pth')
thisdict = {
  "0": "Angry",
  "1": "Bored",
  "2": "Focused",
   "3" :"neutral"
}
print("Predicted class:", thisdict.get(str(predicted_class)))
