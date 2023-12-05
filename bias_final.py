import numpy as np
import torch
from datasetLoader_bias_evalution import DatasetLoader
from MainModel import FacialExpressionCNN, get_class_index_mapping
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

#it is evaluate the fucntion
def EvaluateModel(model, loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for images, labels_batch in loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    return predictions, labels

def analyze_bias(model, TestLoaders, AttributeName, GroupNames):
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

    for group_name in GroupNames:
        loader = TestLoaders[group_name]

        prediction, label = EvaluateModel(model, loader)
        accuracy, precision, recall, f1_score = CalculateMetrics(label, prediction)

        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1-Score'].append(f1_score)

        print(f"\nBias Analysis for {AttributeName} - {group_name}:\n")
        PrintMetrics(accuracy, precision, recall, f1_score)

    # it is calculate averages
    averages = {
        'Accuracy': np.mean(metrics['Accuracy']),
        'Precision': np.mean(metrics['Precision']),
        'Recall': np.mean(metrics['Recall']),
        'F1-Score': np.mean(metrics['F1-Score'])
    }

    print(f"\nOverall Average for {AttributeName}:\n")
    PrintMetrics(averages['Accuracy'], averages['Precision'], averages['Recall'], averages['F1-Score'])

    return averages

def PrintMetrics(accuracy, precision, recall, f1_score):
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

def CalculateMetrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, precision, recall, f1_score


def main():
    # Load the pre-trained model
    ModelPath = '/Users/patelhetulvinodbhai/Desktop/AI2/FacialExpressionModel_best.pth'
    loaded_model = FacialExpressionCNN()
    loaded_model.load_state_dict(torch.load(ModelPath))
    loaded_model.eval()

    # Load class index mapping
    class_index_mapping = get_class_index_mapping('FinalDataset')

    # Load the test loader
    _, _, test_loader = DatasetLoader()

    # Define class names and attributes for bias analysis
    AgeGroups = ['Young', 'Middleage', 'Senior']
    GenderGroups = ['Male', 'Female']

    # Separate the test loader for each age group
    AgeTestLoaders = {
        'Young': test_loader,
        'Middleage': DatasetLoader(root_folder='BIAS_DB/Age/Middleage')[2],
        'Senior': DatasetLoader(root_folder='BIAS_DB/Age/Senior')[2],
    }

    # Separate the test loader for each gender group
    GenderTestLoaders = {
        'Male': DatasetLoader(root_folder='BIAS_DB/Gender/Male')[2],
        'Female': DatasetLoader(root_folder='BIAS_DB/Gender/Female')[2],
    }
    AgeMetrics = analyze_bias(loaded_model, AgeTestLoaders, 'Age', AgeGroups)

    # Perform bias analysis for gender
    GenderMetrics = analyze_bias(loaded_model, GenderTestLoaders, 'Gender', GenderGroups)

    print("Bias Analysis for Age:\n", AgeMetrics)
    print("\nBias Analysis for Gender:\n", GenderMetrics)

    OverallAgeMetrics = {
        'Accuracy': AgeMetrics['Accuracy'],
        'Precision': AgeMetrics['Precision'],
        'Recall': AgeMetrics['Recall'],
        'F1-Score': AgeMetrics['F1-Score']
    }

    OverallGenderMetrics = {
        'Accuracy': GenderMetrics['Accuracy'],
        'Precision': GenderMetrics['Precision'],
        'Recall': GenderMetrics['Recall'],
        'F1-Score': GenderMetrics['F1-Score']
    }

    print("\nOverall Averages for Age:\n")
    PrintMetrics(
        OverallAgeMetrics['Accuracy'],
        OverallAgeMetrics['Precision'],
        OverallAgeMetrics['Recall'],
        OverallAgeMetrics['F1-Score']
    )

    print("\nOverall Averages for Gender:\n")
    PrintMetrics(
        OverallGenderMetrics['Accuracy'],
        OverallGenderMetrics['Precision'],
        OverallGenderMetrics['Recall'],
        OverallGenderMetrics['F1-Score']
    )
    print("\nOverall Averages for the Entire System:\n")
    PrintMetrics(
        overall_metrics['Accuracy'],
        overall_metrics['Precision'],
        overall_metrics['Recall'],
        overall_metrics['F1-Score']
    )


if __name__ == "__main__":
    main()
