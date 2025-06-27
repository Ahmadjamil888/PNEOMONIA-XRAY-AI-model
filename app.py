# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Flask app setup
app = Flask(__name__)
CORS(app)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model/covid_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Labels
class_names = ['NORMAL', 'PNEUMONIA']

@app.route('/predict', methods=['POST'])
def predict():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['xray']
    try:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = class_names[predicted.item()]

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
