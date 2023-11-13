from PIL import Image
import torchvision.transforms as transforms
import torch
import Model_train as mt

# Function to load the model
def load_model(model_path):
    model = mt.FacialExpressionCNN()
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
    ])

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

model = load_model('facial_expression_model.pth')
image_path = 'Dataset/412.jpg'
predicted_class = predict_image_class(image_path, model)
print(f'Predicted class: {predicted_class}')

# Load class index mapping
class_index_mapping = mt.get_class_index_mapping('Dataset')

# Predict and interpret the class
predicted_class_index = predict_image_class(image_path, model)
predicted_class_name = [name for name, index in class_index_mapping.items() if index == predicted_class_index][0]

print(f'Predicted class index: {predicted_class_index}')
print(f'Predicted class name: {predicted_class_name}')