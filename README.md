# 🌸 Flower Classification Web App

This is a Streamlit-based web application that classifies images of flowers using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras.

## 📷 Features
- Upload an image or use your camera to take a photo.
- The model predicts whether the image contains a flower and identifies its type.
- Five supported flower classes:
  - Daisy 🌼
  - Dandelion 🌻
  - Rose 🌹
  - Sunflower 🌞
  - Tulip 🌷
- Confidence threshold: 80%. If prediction confidence is below that, it's marked as **not confidently detected**.

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🖼️ Web Deployment
Deploy on [Streamlit Cloud](https://streamlit.io/cloud) by linking this repo and selecting `app.py` as the main file.

## 🛠️ Files
- `app.py`: Main Streamlit app
- `requirements.txt`: Dependencies
- `Flower_Recog_Model.h5`: Pre-trained CNN model

---

Created with ❤️ by Faisal