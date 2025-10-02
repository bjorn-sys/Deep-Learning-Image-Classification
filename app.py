# ============================================================
# 🌥️ CLOUD CLASSIFIER STREAMLIT APP
# ============================================================

# --- Basic Imports ---
import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

# ============================================================
# 1. PAGE CONFIGURATION (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="🌥️ Cloud Classifier",
    layout="centered"
)

# ============================================================
# 2. MODEL DEFINITION (must match training)
# ============================================================
class Net(nn.Module):
    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================
# Change class names to match your dataset
CLASS_NAMES = [
    "cirriform clouds",
    "cumuliform clouds",
    "stratiform clouds",
    "cumulonimbus clouds",
    "altostratus clouds",
    "cirrocumulus clouds",
    "nimbostratus clouds"
]

MODEL_PATH = "cloud_classifier.pth"   # Make sure this file exists in the same folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(num_classes=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
else:
    st.error(f"❗ Model file `{MODEL_PATH}` not found. Please upload it.")
    st.stop()

# ============================================================
# 4. DEFINE TRANSFORMS (match training transforms)
# ============================================================
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================
def predict_image(image: Image.Image):
    """Predict class for a single uploaded image."""
    image = image.convert("RGB")
    img_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    return CLASS_NAMES[class_idx]

# ============================================================
# 6. STREAMLIT UI
# ============================================================
st.title("🌥️ Cloud Classifier")
st.write("Upload a cloud image to classify its type.")

uploaded_file = st.file_uploader("📤 Upload a cloud image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    with st.spinner("🔍 Classifying..."):
        prediction = predict_image(image)

    st.success(f"✅ Predicted Class: **{prediction}** 🌤️")
else:
    st.info("Please upload an image file to begin.")

# ============================================================
# END OF FILE
# ============================================================
