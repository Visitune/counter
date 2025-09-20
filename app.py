import streamlit as st
from PIL import Image

st.set_page_config(page_title="Test Streamlit Cloud", layout="wide")
st.title("✅ Test Streamlit : upload & affichage")

up = st.file_uploader("Charge une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up).convert("RGB")
    st.image(img, use_column_width=True, caption="Image chargée")
else:
    st.info("Dépose une image ci-dessus.")
