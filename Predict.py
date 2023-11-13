from PIL import Image
import torchvision.transforms as transforms
import torch
import Model_train as mt

# Function to load the model
def load_model(model_path):
    model = mt.FacialExpressionCNN()  # Replace with your model class
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict the class of an image
def predict_image_class(image_path, model):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        # If you used normalization during training, include it here as well
        # transforms.Normalize(mean=[mean_values], std=[std_values])
    ])

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Example usage
model = load_model('path_to_your_model.pth')  # Replace with your model's file path
image_path = 'path_to_your_test_image.jpg'    # Replace with your test image path
predicted_class = predict_image_class(image_path, model)
print(f'Predicted class: {predicted_class}')
