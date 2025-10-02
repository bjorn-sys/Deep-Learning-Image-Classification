import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="☁️ Cloud Image Classifier",
    page_icon="☁️",
    layout="centered"
)

st.title("☁️ Cloud Image Classifier")
st.write("Upload a cloud image and discover its type, along with its natural characteristics.")

# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    # Using ResNet18 backbone
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 4)  # Adjust to 4 cloud classes

    # Try loading your custom trained model
    try:
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
        st.success("✅ Custom model loaded successfully!")
    except Exception as e:
        st.warning("⚠️ Using default pretrained model. (Upload 'model.pth' for better accuracy)")
        st.write(e)

    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# 3. IMAGE TRANSFORMATIONS
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# --------------------------------------------------
# 4. CLOUD CLASSES & DESCRIPTIONS
# --------------------------------------------------
cloud_info = {
    "Cumulus": {
        "description": "Fluffy, white cotton-like clouds often seen on sunny days.",
        "properties": [
            "Low altitude (below 2,000 m)",
            "Indicates fair weather",
            "Flat base with rounded tops"
        ]
    },
    "Cirrus": {
        "description": "Thin, wispy clouds high up in the sky.",
        "properties": [
            "High altitude (above 6,000 m)",
            "Made mostly of ice crystals",
            "Indicates approaching changes in weather"
        ]
    },
    "Stratus": {
        "description": "Grayish, uniform clouds that often cover the whole sky like a blanket.",
        "properties": [
            "Low altitude (below 2,000 m)",
            "Often brings light drizzle or mist",
            "Appears as a dull, overcast layer"
        ]
    },
    "Nimbus": {
        "description": "Dark, dense rain-bearing clouds.",
        "properties": [
            "Low to middle altitude",
            "Associated with heavy rain or storms",
            "Thick and towering appearance"
        ]
    }
}

class_names = list(cloud_info.keys())

# --------------------------------------------------
# 5. PREDICTION FUNCTION
# --------------------------------------------------
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    predicted_class = class_names[predicted.item()]
    return predicted_class, confidence

# --------------------------------------------------
# 6. IMAGE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a cloud image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🔎 Classify Cloud"):
        with st.spinner("⏳ Classifying the cloud type..."):
            label, confidence = predict_image(image)

        # Display prediction
        st.success(f"✅ Predicted Type: **{label}**")
        st.info(f"📈 Confidence: **{confidence*100:.2f}%**")

        # Show explanation
        st.markdown("### 🌥️ Cloud Type Details")
        st.markdown(f"**Definition:** {cloud_info[label]['description']}")
        st.markdown("**Key Properties:**")
        for prop in cloud_info[label]['properties']:
            st.markdown(f"- {prop}")

        st.markdown("---")
        st.caption("💡 The model considers shape, texture, and color patterns to identify cloud types. For example, fluffy edges and bright tops often point to **Cumulus**, while layered gray patterns suggest **Stratus**.")
else:
    st.info("⬆️ Please upload a cloud image to get started.")

# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by PyTorch • Streamlit • ResNet18")
