# AIducation
Facial Expression detection for the Education System

## File Ak_9.pdf:
This File is the Report file for the Project Delieverable 1

## File Dataset_origin.pdf:
A file or document detailing the provenance of each dataset/image:

## Description of datapreprcessing.py
### Features

1. Setting Up Directory Paths:
   - Fetches the current working directory and constructs paths for the dataset.
2. Verifying Dataset Directory:
   - Ensures the dataset directory is present.
3. Image Categories:
   - Processes images under predefined categories: "Neutral", "bored", "focused" and "angry".
4. Fetching Images:
   - Retrieves all .jpg or .png images from the specified directory.
5. Resizing and Grayscale Conversion:
   - Images are resized to a consistent size (64x64 pixels).
   - Converts the resized images to grayscale.
6. Processing & Saving:
   - For each category, the script resizes, converts, and saves grayscale images to a newly created directory.
7. Visualizations:
   - Displays grayscale images in a 5x5 grid.
   - Presents histograms for pixel intensities of randomly selected images.
   - Shows a bar chart reflecting the distribution of images across categories.

### Usage

- Ensure you have the required libraries: os, cv2, matplotlib, and random.
- Place your dataset in a directory named "Cleaning" with subdirectories corresponding to the categories.
- Run the datapreprocessing.py script as given below.

### Steps for running code file datapreprocessing.py:

1. Create the new folder and it contains the one folder named “Cleaning” and one python file called datapreprocessing.py.
2. Inside the “Cleaning” ,there must be all four category  with train database like “angry”->all images.
3. After that run the datapreprocessing.py file.
4. Now, programs gives the output of all the things about pixel intensity,bar graph and so on.
5. And it also Crete the outputFile for all category where all image is stored by category so users can compared with original database like function like resize ,grayscale and so on.

## File Originality_form.pdf
A form that has been signed by all the team memebers for the Originality of Work.
