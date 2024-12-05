import streamlit as st
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import torch.nn as nn

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AGRIVISION",
    page_icon="C:/Users/Senku Ishigami/Downloads/AGRIVISION_LOGO-removebg-preview.png",
    layout="centered"
)

# Import custom font for title
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    </style>
    """,
    unsafe_allow_html=True
)

# --- Streamlit Centered Logo and Title ---
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; text-align: center; margin-top: 20px;">
        <img src="data:image/png;base64,{base64.b64encode(open('C:/Users/Senku Ishigami/Downloads/AGRIVISION_LOGO-removebg-preview.png', 'rb').read()).decode()}" 
             alt="AGRIVISION Logo" 
             style="width: 150px; height: 150px; margin-right: 20px;">
        <h1 style="
            font-family: 'Orbitron', sans-serif; 
            font-size: 50px; 
            color: #ffffff; 
            background: linear-gradient(90deg, #FFD700, #FF8C00, #FF4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 4px 4px rgba(0, 0, 0, 0.5);
            margin: 0;
        ">
            AGRIVISION
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for Model Selection ---
st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{base64.b64encode(open('C:/Users/Senku Ishigami/Downloads/AGRIVISION_LOGO-removebg-preview.png', 'rb').read()).decode()}" 
             alt="AGRIVISION Logo" 
             style="width: 100px; height: auto; margin-bottom: 20px;">
    </div>
    <h2 style="
        font-family: 'Orbitron', sans-serif; 
        font-size: 24px; 
        color: #ffffff; 
        background: linear-gradient(90deg, #00ff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 4px 4px rgba(0, 0, 0, 0.5);
        text-align: center; 
        margin-bottom: 20px;
    ">
        🌿 Controls
    </h2>
    """,
    unsafe_allow_html=True
)

# Model selection dropdown
model_choice = st.sidebar.selectbox("Choose Model", ["GCN", "SLIC"])
alpha = st.sidebar.slider("Overlay Transparency", 0.1, 1.0, 0.5)

  #--- Model Definitions ---
class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        # Define your GraphSAGE architecture here (simplified)
        self.fc = nn.Linear(128, 1)  # Adjust based on your model's output

    def forward(self, x):
        # Forward pass logic here
        return self.fc(x)

# --- Load U-Net Model ---
@st.cache_resource
def load_unet_model():
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(
        "C:/Users/Senku Ishigami/Documents/Assignment-elective-main/unet_resnet50epoch_model.pth",
        map_location=torch.device('cpu'),
        weights_only=True  # Ensures only weights are loaded, not arbitrary code
    ))
    model.eval()
    return model

# --- Load GraphSAGE Model ---
@st.cache_resource
def load_graphsage_model():
    model = GraphSAGE()  # Initialize the GraphSAGE model here
    checkpoint = torch.load(
        "C:/Users/Senku Ishigami/Documents/Assignment-elective-main/GraphSageModel.pth",
        map_location=torch.device('cpu'),
        weights_only=True  # Ensures only weights are loaded
    )
    model.load_state_dict(checkpoint['model_state_dict'])  # Correctly extract state_dict
    model.eval()
    return model

# --- Sidebar for Model Selection ---
st.sidebar.markdown("### Select Model:")
model_choice = st.sidebar.radio("Choose a model:", ("GCN", "SLIC"))

# Initialize model only after user selection
if model_choice == "GCN":
    model = load_unet_model()
elif model_choice == "SLIC":
    model = load_graphsage_model()

# --- Set Background ---
# Function to set background with updated styles
def add_bg_from_local(file_path):
    with open(file_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
                background-size: cover;
            }}
            .stButton>button {{
                background-color: #31511E;
                color: #F6FCDF;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #859F3D;
            }}
            .stButton>button:hover {{
                background-color: #859F3D;
                border-color: #F6FCDF;
                color: #1A1A19;
            }}
            h1, h2, h3 {{
                color: #F6FCDF;
                text-align: center;
                font-family: 'Orbitron', sans-serif;
                text-shadow: 3px 3px #1A1A19;
            }}
            .stMarkdown p {{
                color: #F6FCDF;
                font-size: 14px;
                font-family: 'Orbitron', sans-serif;
                text-shadow: 1px 1px #1A1A19;
            }}
            .uploadedFile {{
                background-color: #31511E;
                color: #FFA500;
                border-radius: 8px;
            }}
            .stSidebar {{
                background-color: rgba(26, 26, 25, 0.9);
                color: #F6FCDF;
                padding: 20px;
                border-radius: 10px;
            }}
            .stSidebar .stFileUploader {{
                background-color: #1A1A19;
                border: 2px solid #859F3D;
                border-radius: 8px;
            }}
            .stSlider {{
                color: #F6FCDF;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

background_image_path = "C:/Users/Senku Ishigami/Downloads/123233.jpg"
if os.path.exists(background_image_path):
    add_bg_from_local(background_image_path)
else:
    st.warning("Background image not found. Ensure the file is in your dataset directory.")

# --- Main Section ---
st.markdown("**Upload an image to classify agricultural areas!**")
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

# --- Main Section ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess image for U-Net or GraphSAGE (adjust based on GraphSAGE input requirements)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
        prediction_resized = Image.fromarray((prediction * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)

    # Display input and prediction side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(prediction_resized, alpha=alpha, cmap='Greens')
    
    # Dynamically change title based on the selected model
    axes[1].set_title("Predicted Overlay")  

    
    axes[1].axis('off')
    
    st.pyplot(fig)
    
    
    # Descriptive text for results
    st.markdown("### 📝 Result Description:")
    st.markdown("""
    The green areas in the image represent the detected agricultural zones. These regions are classified based on the model's analysis of the uploaded image. The overlay helps highlight areas of interest for further examination or planning.
    """)

    # Download button for predicted mask
    prediction_resized.save("predicted_mask.png")
    with open("predicted_mask.png", "rb") as file:
        st.download_button(label="📥 Download Predicted Mask", data=file, file_name="predicted_mask.png", mime="image/png")



