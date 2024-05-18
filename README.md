# American Sign Language Real-Time Detection

## Description

This project aims to facilitate real-time detection of American Sign Language (ASL) gestures using computer vision techniques. ASL is a crucial mode of communication for individuals with hearing impairments, and automating its recognition can significantly improve accessibility. The dataset utilized in this project, available on Kaggle, consists of images capturing ASL gestures representing letters from A to Z. The primary objective is to develop a system capable of recognizing these gestures in real-time and mapping them to their corresponding English alphabet letters.

## Dataset

The ASL Alphabet Dataset employed for this project can be accessed on Kaggle via the following link: [ASL Alphabet Dataset](https://www.kaggle.com/code/alfathterry/american-sign-language-real-time-detection). This dataset comprises images of ASL gestures for each letter of the alphabet, along with their respective labels.

## Project Components

1. **Preprocessing**
   - **Purpose:** The preprocessing stage involves preparing the dataset for training by extracting relevant features from the images.
   - **Script:** `preprocessing.py`
   - **Details:** 
     - Load the ASL Alphabet Dataset from Kaggle.
     - Organize the dataset into a structured directory format suitable for processing. For instance, group images for each letter into separate folders.
     - Implement image loading and preprocessing techniques.
     - Utilize the MediaPipe library to extract hand landmarks from the images.
     - Transform the extracted data into a suitable format for machine learning.
     - Save the preprocessed data as `data.pickle` for subsequent training.

2. **Training**
   - **Purpose:** Train a machine learning model on the preprocessed data to recognize ASL gestures.
   - **Script:** `training.py`
   - **Details:**
     - Load the preprocessed data from `data.pickle`.
     - Split the data into training and testing sets using `train_test_split`.
     - Initialize a Random Forest Classifier or another suitable machine learning model.
     - Train the model using the training data and evaluate its performance.
     - Save the trained model as `model.p` for real-time detection.

3. **Real-Time Detection**
   - **Purpose:** Develop a real-time ASL gesture recognition system using the trained model.
   - **Script:** `main.py`
   - **Details:**
     - Load the trained machine learning model from `model.p`.
     - Initialize the MediaPipe Hands model for hand landmark detection.
     - Capture video frames from the webcam using OpenCV.
     - Process each frame to detect hand landmarks using MediaPipe.
     - Utilize the trained model to predict the ASL gesture based on the detected hand landmarks.
     - Display the real-time video stream with overlaid predicted ASL gesture on the screen.

## Usage

1. **Dataset Preparation**
   - Download the ASL Alphabet Dataset from Kaggle using the provided link.
   - Extract and organize the dataset into a structured format, grouping images by letter gestures.

2. **Preprocessing**
   - Execute the `processing.py` script to preprocess the dataset and extract hand landmarks.
   - Ensure the data is transformed into a suitable format for training and save it as `data.pickle`.

3. **Training**
   - Run the `training.py` script to train the machine learning model on the preprocessed data.
   - Evaluate the model's performance and save it as `model.p` for real-time detection.

4. **Real-Time Detection**
   - Execute the `main.py` script to perform real-time ASL gesture detection using the trained model.
   - Ensure the webcam is accessible and properly configured to capture video frames.
   - Observe the real-time video stream with overlaid ASL gesture predictions for each detected hand gesture.

## Dependencies

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- pickle

## References

- [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/code/alfathterry/american-sign-language-real-time-detection)
- MediaPipe Library Documentation
- scikit-learn Documentation

**Note:** Adjust file paths, directory structures, and dependencies as required based on your local environment and setup. Ensure all necessary libraries are installed and configured correctly to run the scripts smoothly.
