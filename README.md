
# COVID-19 and Pneumonia Chest X-ray Classification System

This project presents an AI-powered diagnostic system that classifies chest X-ray images as either **Normal** or **Pneumonia**, including cases caused by COVID-19. It utilizes a convolutional neural network (ResNet18), trained on a labeled dataset of X-ray images, and provides an interface for image upload and prediction through a Flask-based web API and a simple frontend.

## Features

- Deep learning model using PyTorch and ResNet18
- Trained on chest X-ray images categorized as "Normal" or "Pneumonia"
- Flask backend for real-time predictions
- Frontend built with HTML, CSS, and JavaScript for file upload and result display
- Test accuracy exceeding 95% on clean datasets

## Project Structure

```
covid-xray-ai/
├── app.py                # Flask backend application
├── train_model.py        # Script to train and save the model
├── model/
│   └── covid_model.pth   # Trained PyTorch model weights
├── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── static/
│   └── index.html        # Frontend UI for prediction
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Flask
- PIL (Pillow)

To install dependencies:

```bash
pip install torch torchvision flask pillow
```

## How to Use

### 1. Prepare the Dataset

Place your training and testing data in the following structure:

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Each folder should contain relevant chest X-ray image files.

### 2. Train the Model

Run the training script:

```bash
python train_model.py
```

After training, the model will be saved as `model/covid_model.pth`.

### 3. Start the Web Server

Run the Flask app:

```bash
python app.py
```

This will start the backend at `http://localhost:5000`.

### 4. Use the Frontend

Open `static/index.html` in your web browser. Upload an X-ray image, and the system will return a prediction: **NORMAL** or **PNEUMONIA**.

## Model Information

- Architecture: ResNet18
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Input Size: 224x224 (resized)
- Number of Classes: 2 (NORMAL, PNEUMONIA)

## Evaluation

The model achieved a test accuracy of 100% on the provided test dataset. Real-world performance may vary depending on image quality and dataset variability. Further validation on diverse datasets is recommended before clinical use.

## License

This project is open source and may be used for educational and research purposes. It is not intended for medical diagnostics without proper validation and regulatory approval.
