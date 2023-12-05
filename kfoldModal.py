import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  random_split, Subset, SubsetRandomSampler, Dataset
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from datasetLoader import dataset_loader
from evaluation import evalutionModel, printing, ConfusionMatrixPlotting
# Assuming FacialExpressionCNN is your model class
from MainModel import FacialExpressionCNN, get_class_index_mapping
def k_fold_cross_validation(model, loader, k):
    AllTestPredictions = []
    AllTestLabels = []

    # Extract class labels from the dataset
    ActualLabels = [loader.dataset[i][1] for i in range(len(loader.dataset))]
    print("Actual Class Labels:", set(ActualLabels))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (_, test_indices) in enumerate(skf.split(range(len(loader.dataset)), ActualLabels)):
        print(f"Fold {fold + 1}/{k}")
        # Set the model to training mode
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        epochs = 10
        for epoch in range(epochs):
            RunningLoss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                RunningLoss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Epoch {epoch + 1} loss: {RunningLoss / len(loader)}, Accuracy: {accuracy * 100}%')

        model.eval()
        TestPredictions, TestLabels = evalutionModel(model, loader)
        class_names = ["Angry", "Bored", "Focused", "Neutral"]
        # Calculating metrics for the current fold
        printing(TestLabels, TestPredictions, class_names)

        AllTestPredictions.extend(TestPredictions)
        AllTestLabels.extend(TestLabels)

    return AllTestPredictions, AllTestLabels

#it is the fucntion that take whole dataset
class WholeDataset(Dataset):
    def __init__(self, root_folder='FinalDataset', transform=None):
        self.dataset = datasets.ImageFolder(root=root_folder, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

#it is load the fucntion that load the data
def DatasetLoader(root_folder='FinalDataset', batch_size=32):
    DataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    whole_dataset = WholeDataset(root_folder=root_folder, transform=DataTransform)
    AllLoading = DataLoader(whole_dataset, batch_size=batch_size, shuffle=True)

    return AllLoading
def main():
    batch_size = 32
    AllLoading = DatasetLoader(batch_size=batch_size)

    ModalPath = '/Users/patelhetulvinodbhai/Desktop/AI2/FacialExpressionModel_best.pth'
    loaded_model = FacialExpressionCNN()
    loaded_model.load_state_dict(torch.load(ModalPath))
    loaded_model.eval()
    class_names = ["Angry", "Bored", "Focused", "Neutral"]
    ClassIndexMapping = get_class_index_mapping('FinalDataset')

    # Perform k-fold cross-validation on the test set
    k = 10
    TestPredictions, TestLabels = k_fold_cross_validation(loaded_model, AllLoading, k)
    print("Predictions:", TestPredictions)
    print("Labels:", TestLabels)

    # Plot confusion matrix for the aggregated test set
    ConfusionMatrixPlotting(TestLabels, TestPredictions, list(ClassIndexMapping.keys()))

    # Print metrics for the aggregated test set
    print("\nMetrics for Aggregated Test Set (K-fold Cross-Validation):\n")
    print("=" * 40)
    printing(TestLabels, TestPredictions, class_names)

if __name__ == "__main__":
    main()
