# Character Classification using PyTorch

This repository contains a PyTorch-based Convolutional Neural Network (CNN) designed to classify images of characters into one of 62 classes: 
- 10 digits (0-9)
- 26 uppercase letters (A-Z)
- 26 lowercase letters (a-z)

## Project Structure

* `model.py`: Defines the `CharacterClassification` CNN architecture.
* `train.py`: Handles data loading, preprocessing, data augmentation, and the training/evaluation loops. 
* `test.py`: An inference script that loads the trained model, processes an external image, predicts the character, and displays the result visually using Matplotlib.

## Requirements

To run this project, you will need Python 3.x and the following libraries installed:

* `torch`
* `torchvision`
* `Pillow`
* `matplotlib`

## Dataset Structure

This project is based on the **Chars74K dataset** (specifically the English fonts subset).

The `train.py` script uses `torchvision.datasets.ImageFolder`. It expects your dataset to be located at `data/EnglishFnt/English/Fnt` relative to the script's execution path. The dataset should be structured so that each of the 62 classes has its own subdirectory containing the respective image samples.

You can download the dataset from the official Chars74K website or Kaggle and extract it into the `data/` directory.

## Usage

### 1. Training the Model
Run the `train.py` script to begin training. By default, it splits the dataset into 80% training and 20% testing data. It also applies data augmentations like random rotations and translations to the training set to prevent overfitting.

```bash
python train.py
```
Once training is complete, the script saves the trained model weights as `model.pth` in the current directory.

### 2. Testing and Inference
To predict the character in a new image, place your image in the root directory (the script defaults to looking for `test_image.png`), or update the `image_path` variable inside `test.py`.

Run the testing script:
```bash
python test.py
```

The console will output the predicted class index, the character, and the model's confidence percentage. It will also open a Matplotlib window displaying the image alongside the visual prediction.