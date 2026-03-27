# 🖍️ Handwritten Digit Recognizer (CNN)

## 🚀 Overview
This project is an end-to-end Machine Learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch.

It includes:
- Model training pipeline
- Evaluation and visualization
- Interactive web app using Streamlit

---

## 🧠 Problem Statement
Handwritten digit recognition is a classic problem in computer vision. The challenge is to correctly classify images of digits despite variations in handwriting styles.

---

## ⚙️ Approach

### Model: Convolutional Neural Network (CNN)

Instead of using a simple neural network (MLP), this project uses a CNN to preserve spatial relationships in images.

Architecture:
- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Fully Connected Layers
- Output (10 classes)

---

## 📊 Results

- Test Accuracy: ~97–99%
- Improved performance compared to basic MLP

---

## 🖥️ Demo (Streamlit App)

The app allows users to:
- Upload an image of a handwritten digit
- View processed 28×28 input
- Get prediction with confidence scores

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Streamlit
- Plotly
- NumPy / Pandas

---

## 📁 Project Structure

mnist-project/
│
├── src/
│ ├── model.py
│ ├── train.py
│
├── app.py
├── requirements.txt
├── README.md


---

## ▶️ How to Run

### 1. Clone the repository

git clone https://github.com/SamarthMantri8665/mnist-digit-recognizer.git
cd mnist-project


### 2. Install dependencies

pip install -r requirements.txt


### 3. Train the model

python -m src.train


### 4. Run the app

streamlit run app.py



---

## 📌 Key Learnings

- Importance of CNN over MLP for image tasks
- Model training and evaluation pipeline
- Building interactive ML apps with Streamlit
- Structuring ML projects properly

---

## 🔮 Future Improvements

- Add drawing canvas for real-time digit input
- Compare CNN vs MLP performance visually
- Deploy the app online
- Improve model with deeper architectures

---

## 🤝 Contribution

Feel free to fork this project and improve it!

---

