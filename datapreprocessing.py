import os
import cv2
import matplotlib.pyplot as plt
import random
# Getting the current working directory
FetchCurrentDir = os.getcwd()

# It define the dataset directory name
NameOfDirectoryDatasetDir = "Cleaning"

# It constructs the full dataset directory path
DatasetDir = os.path.join(FetchCurrentDir, NameOfDirectoryDatasetDir)

# Check if the dataset directory exists
if not os.path.exists(DatasetDir):
    print(f"Dataset directory not found: {DatasetDir}")
    exit()

# It is define the categories
categories = ["Neutral","bored","angry","focused"]

# It is the function that go through images from a database
def GetImage(EmotionDir):
    images = []
    for filename in os.listdir(EmotionDir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(EmotionDir, filename))
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to Get Image: {os.path.join(EmotionDir, filename)}")
    return images

# It is the function for resize and convert images to grayscale
def ResizeAndGrayscale(images, ExpectedSize):
    NewResizedImages = []
    for img in images:
        NewResizedImg = cv2.resize(img, ExpectedSize)
        
        # It is for checking if the image has been resized successfully or not.
        if NewResizedImg.shape[:2] != ExpectedSize:
            print(f"Failed to resize an image to the expected size: {img.shape} -> {ExpectedSize}")
        
        # Then,It helps to convert the resized image to grayscale
        GrayscaleImg = cv2.cvtColor(NewResizedImg, cv2.COLOR_BGR2GRAY)
        
        NewResizedImages.append(GrayscaleImg)
    
    return NewResizedImages

# It is the Desired size for resizing
ExpectedSize = (48, 48) 
TotalImageDisplay = 25
Counter = {}

# Iterate through each categories and call upper fucntion for data cleaning
for category in categories:
    EmotionDir = os.path.join(DatasetDir, category)
    
    # Getting the images from the each category folder
    images = GetImage(EmotionDir)
    
    # It counts the number of images in the each category
    TotalnumberImage = len(images)
    print(f"Category: {category}, Number of Images: {TotalnumberImage}")
    
    # It helps to resize and convert images to grayscale in  category
    FinalGrayscaleImages = ResizeAndGrayscale(images, ExpectedSize)
    
    # It creates a new directory for grayscale images
    ResultOutputDir = os.path.join(DatasetDir, f"OutputFile{category}")
    os.makedirs(ResultOutputDir, exist_ok=True)
    
    # It saves grayscale images to the category's directory
    for i, grayscaleImage in enumerate(FinalGrayscaleImages):
        ResultOutputFilename = os.path.join(ResultOutputDir, f"{i + 1}.jpg")
        cv2.imwrite(ResultOutputFilename, grayscaleImage)
    
    RandomImages = random.sample(FinalGrayscaleImages, min(TotalImageDisplay, len(FinalGrayscaleImages)))
    
    # Create a subplot of 5x5 images as per the project document
    plt.figure(figsize=(7, 7))
    plt.suptitle(category, fontsize=16) 
    for i in range(len(RandomImages)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(RandomImages[i], cmap='gray')
        plt.axis('off')
    
    # Display the images for this category
    plt.show()
    Counter[category] = TotalnumberImage

    plt.figure(figsize=(12, 9))
    for i, random_image in enumerate(RandomImages):
        plt.subplot(5, 5, i + 1)
        plt.hist(random_image.ravel(), bins=256, range=(0, 256), density=True, alpha=0.6, color='k', label='Total Intensity')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
    
    # Display the histograms for this category
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 5))
plt.bar(Counter.keys(), Counter.values())
plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.show()

