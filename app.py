import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import plotly.express as px

import pandas as pd
from src.model import CNNModel


# 1. Page Configuration
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🖍️", layout="centered")
st.title("🖍️ Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9), and the PyTorch model will guess what it is!")

# 2. Define the Model Architecture (Must match exactly what was trained)
model = CNNModel()

# 3. Load the Model Weights
@st.cache_resource
def load_model():
    model = CNNModel()
    # Load the saved weights from yesterday's lab
    model.load_state_dict(torch.load('mnist_lightweight_model.pth', weights_only=True))
    model.eval() # Set to evaluation mode
    return model

model = load_model()

# 4. Image Preprocessing Pipeline
def process_image(image, invert_colors):
    # Convert to grayscale
    img = image.convert('L')
    
    # MNIST is white digits on black background. 
    # If user uploads black text on white paper, we MUST invert it.
    if invert_colors:
        img = ImageOps.invert(img)
        
    # Resize to the 28x28 format the model expects
    img = img.resize((28, 28))
    
    # Convert to PyTorch Tensor and add batch dimension (1, 1, 28, 28)
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    return tensor, img

# 5. User Interface: File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add a helpful checkbox for the color inversion problem
invert_colors = st.checkbox("Invert Colors (Check this if your image is black ink on a white background)", value=True)

# 6. Prediction Engine
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    raw_image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Image")
        st.image(raw_image, use_container_width=True)
        
    # Process the image
    tensor, processed_img = process_image(raw_image, invert_colors)
    
    with col2:
        st.subheader("What the Model Sees (28x28)")
        # Resize back up just so it's visible on the web page
        st.image(processed_img.resize((150, 150)), use_container_width=False)

    st.markdown("---")
    
    # Run the model
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
        prediction = torch.argmax(outputs, dim=1).item()

    # Display the final answer
    st.success(f"### The model predicts this is a: **{prediction}**")
    
    # 7. Beautiful Interactive Confidence Chart
    st.subheader("Model Confidence Breakdown")
    
    # Create a dataframe for Plotly
    df_probs = pd.DataFrame({
        'Digit': [str(i) for i in range(10)],
        'Probability': probabilities * 100
    })
    
    # Plotly Bar Chart
    fig = px.bar(
        df_probs, 
        x='Digit', 
        y='Probability',
        text_auto='.2f',
        title="Probability Distribution across all digits",
        labels={'Probability': 'Confidence (%)'},
        color='Probability',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload an image file to get started.")
