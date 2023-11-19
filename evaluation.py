import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import variantTwo as mt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# It is the function to load the function
def DatasetLoading(RootDirectory, Splitting):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
    ])

    CompleteDataset = datasets.ImageFolder(root=RootDirectory, transform=transform)

    #it is the code for splitting the dataset according to assignmnet by the given ratio
    TranningSize = int(Splitting[0] * len(CompleteDataset))
    ValidationSize = int(Splitting[1] * len(CompleteDataset))
    TestSize = len(CompleteDataset) - TranningSize - ValidationSize

    TrainDataset, lastDataset = torch.utils.data.random_split(CompleteDataset,
                                                                     [TranningSize, len(CompleteDataset) - TranningSize])
    ValidationDataset, TestDataset = torch.utils.data.random_split(lastDataset, [ValidationSize, TestSize])

    return TrainDataset, ValidationDataset, TestDataset

#it is the fucntion to evalute the model.
def evalutionModel(model, loader):
    model.eval()
    prediction = []
    label = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            prediction.extend(predicted.cpu().numpy())
            label.extend(labels.cpu().numpy())

    return prediction, label


# it is the function to generate and plot confusion matrix
def ConfusionMatrixPlotting(yTrue, yPred, class_names):
    cm = confusion_matrix(yTrue, yPred)
    plt.figure(figsize=(len(class_names), len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

#it is basically print the scores
def printing(yTrue, yPred, class_names):
    cm = confusion_matrix(yTrue, yPred)

    print(f"\n{'=' * 40}\n")
    print("Metrics:")
    print("\n{:>20} {:>10} {:>10} {:>10}".format("Class", "Precision", "Recall", "F1-Score"))

    for i in range(len(class_names)):
        precision = cm[i, i] / max(1, sum(cm[:, i]))
        recall = cm[i, i] / max(1, sum(cm[i, :]))
        f1_score = 2 * (precision * recall) / max(1, precision + recall)

        print("{:>20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            class_names[i], precision, recall, f1_score
        ))

        # print("True Positives for {}: {}".format(class_names[i], cm[i, i]))

    # Calculate macro and micro averages
    MacroMetrics = precision_recall_fscore_support(yTrue, yPred, average='macro')
    MicroMetrics = precision_recall_fscore_support(yTrue, yPred, average='micro')

    print("\nMacro Average:")
    print("{:>20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "Macro Average", MacroMetrics[0], MacroMetrics[1], MacroMetrics[2]
    ))

    print("\nMicro Average:")
    print("{:>20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "Micro Average", MicroMetrics[0], MicroMetrics[1], MicroMetrics[2]
    ))

    accuracy = accuracy_score(yTrue, yPred)
    print("\nOverall Accuracy: {:.4f}".format(accuracy))


def main():
    # Define the root directory of your dataset
    RootDirectory = 'provide the path of dataset'

    # Define split ratios
    Splitting = [0.7, 0.15, 0.15]

    # Load the dataset
    TrainDataset, ValidationDataset, TestDataset = DatasetLoading(RootDirectory, Splitting)

    # Create DataLoaders
    batch_size = 32
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=batch_size, shuffle=False)
    TestLoader = DataLoader(TestDataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = mt.FacialExpressionCNN()
    model.load_state_dict(torch.load('variant2.pth'))
    model.eval()

    # Load class index mapping
    ClassIndexMapping = mt.get_class_index_mapping('FinalDataset')

    # Evaluate the model on the training set
    TrainPredictions, TrainLabels = evalutionModel(model, TrainLoader)

    # Plot confusion matrix for training set
    ConfusionMatrixPlotting(TrainLabels, TrainPredictions, list(ClassIndexMapping.keys()))
    class_names = ["angry", "bored", "focused", "neutral"]

    # Print metrics for training set
    print("\nMetrics for Training Set:\n")
    print("=" * 40)
    printing(TrainLabels, TrainPredictions, class_names)

    ValidationPredictions, ValidationLabels = evalutionModel(model, ValidationLoader)

    # Plot confusion matrix for validation set
    ConfusionMatrixPlotting(ValidationLabels, ValidationPredictions, list(ClassIndexMapping.keys()))

    # Print metrics for validation set
    print("\nMetrics for Validation Set:\n")
    print("=" * 40)
    printing(ValidationLabels, ValidationPredictions, class_names)

    TestPredictions, TestLabels = evalutionModel(model, TestLoader)

    # Plot confusion matrix for test set
    ConfusionMatrixPlotting(TestLabels, TestPredictions, list(ClassIndexMapping.keys()))

    # Print metrics for test set
    print("\nMetrics for Test Set:\n")
    print("=" * 40)
    printing(TestLabels, TestPredictions, class_names)


if __name__ == "__main__":
    main()