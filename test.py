import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import torch
import torch.nn.functional as F
from train import transforms
from model import CharacterClassification

#takes an image that is in the same directory as this file and predicts the character in it using the trained model

image_path = 'test_image.png' # Replace with your image path

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the image

img = Image.open(image_path)

# Mapping for 62 classes [0-9], [A-Z], [a-z]
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

 
# Preprocess the image and prepare it for the model using the same transforms as during training 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterClassification().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # Load your trained model

img_input = transform(img).unsqueeze(0).to(device) # Add batch dimension and move to GPU

# Predict the character
model.eval()
with torch.no_grad():
    output = model(img_input)
    probabilities = F.softmax(output, dim=1) # Convert logits to probabilities
    prediction_idx = torch.argmax(probabilities, dim=1).item() # Get the index of the predicted class
    confidence = probabilities[0][prediction_idx].item() # Get the confidence of the prediction

# Log the results
predicted_char = chars[prediction_idx]
print(f"Predicted Class Index: {prediction_idx}")
print(f"Predicted Character: {predicted_char}")
print(f"Confidence: {confidence:.2%}")

# Visual confirmation
plt.imshow(img, cmap='gray')
plt.title(f"Prediction: {predicted_char} ({confidence:.2%})")
plt.axis('off')
plt.show()