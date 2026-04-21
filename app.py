import streamlit as st
from PIL import Image
import pandas as pd
import requests

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Vision Classifier",
    page_icon="🤖",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API CALL ----------------
API_URL = "http://127.0.0.1:8000/predict"

def call_api(image_file):
    try:
        response = requests.post(
            API_URL,
            files={"file": image_file.getvalue()}
        )
        return response.json()
    except:
        return {"error": "API not reachable"}

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
top_k = st.sidebar.slider("Top Predictions", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image and let the AI classify it.")

# ---------------- MAIN ----------------
st.title("🤖 AI Vision Classifier")
st.markdown("Upload an image and get instant AI predictions.")

uploaded_file = st.file_uploader(
    "Drag & drop or upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PROCESS ----------------
if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    # LEFT: IMAGE
    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    # RIGHT: RESULTS
    with col2:
        st.subheader("🧠 Predictions")

        with st.spinner("Analyzing..."):
            result = call_api(uploaded_file)

        if "error" in result:
            st.error(result["error"])

        elif "predictions" in result and len(result["predictions"]) > 0:
            preds = result["predictions"][:top_k]

            df = pd.DataFrame(preds)

            # Best prediction
            best = preds[0]
            st.success(f"**{best['class']}** ({best['confidence']:.2%})")

            # Chart
            st.bar_chart(df.set_index("class"))

            # Table
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("No predictions returned.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI + YOLOv8 + Streamlit")
