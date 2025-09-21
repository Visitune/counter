# app.py

import streamlit as st
from PIL import Image
import torch
from sentence_transformers import util
import utils
import io
import numpy as np

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Module `streamlit-image-coordinates` manquant.")
    st.stop()

st.set_page_config(page_title="Comptage Adaptable", layout="wide")
st.title(" pipette à couleur")

# --- Barre Latérale ---
with st.sidebar:
    st.header("1. Paramètres de Comptage")
    # Initialisation de l'état de la session pour les sliders
    if 'h_min' not in st.session_state:
        st.session_state.update({'h_min': 0, 'h_max': 25, 's_min': 120, 's_max': 255, 'v_min': 120, 'v_max': 255})

    st.subheader("Plage de Couleur (HSV)")
    st.session_state.h_min = st.slider("Teinte Min", 0, 179, st.session_state.h_min)
    st.session_state.h_max = st.slider("Teinte Max", 0, 179, st.session_state.h_max)
    st.session_state.s_min = st.slider("Saturation Min", 0, 255, st.session_state.s_min)
    st.session_state.s_max = st.slider("Saturation Max", 0, 255, st.session_state.s_max)
    st.session_state.v_min = st.slider("Valeur Min", 0, 255, st.session_state.v_min)
    st.session_state.v_max = st.slider("Valeur Max", 0, 255, st.session_state.v_max)
    
    st.subheader("Affinement de la Détection")
    min_area = st.slider("Taille min objet (px²)", 10, 5000, 150)
    selection_radius = st.slider("Rayon de clic (Comptage)", 5, 50, 20)

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

model = utils.get_clip_model()

# Correction majeure de la gestion du cache et de l'état
@st.cache_data
def load_image_data(_img_bytes):
    return Image.open(io.BytesIO(_img_bytes)).convert("RGB")

base_img = load_image_data(up.getvalue())

# Le traitement n'est plus dans le cache global, il dépend des sliders
hsv_params = {'h_min': st.session_state.h_min, 'h_max': st.session_state.h_max, 's_min': st.session_state.s_min, 
              's_max': st.session_state.s_max, 'v_min': st.session_state.v_min, 'v_max': st.session_state.v_max}
candidates, binary_mask, img_hsv = utils.detect_candidates_by_color(base_img, model, min_area, hsv_params)
st.session_state.objects = candidates

# --- Interface Principale ---
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Étape 2 : Calibrer & Guider")
    click_mode = st.radio("Action du clic sur l'image :", [" pipette couleur", "Sélectionner un exemple"])
    
    st.info(f"Mode actif : **{click_mode}**. Cliquez sur l'image à gauche.")
    
    if click_mode == "Sélectionner un exemple":
        text_prompt = st.text_input("Ou guider par texte :", "an object")
        if text_prompt and text_prompt != "an object":
            st.session_state.target_embedding = model.encode(text_prompt, convert_to_tensor=True)
    
    st.header("Étape 3 : Compter")
    similarity_threshold = st.slider("Seuil de similarité", 0.5, 1.0, 0.85, 0.01)
    
    final_count = 0
    if st.session_state.get('target_embedding') is not None and st.session_state.objects:
        all_embeddings = torch.stack([obj['embedding'] for obj in st.session_state.objects])
        cos_scores = util.cos_sim(st.session_state.target_embedding, all_embeddings)[0]
        for i, obj in enumerate(st.session_state.objects):
            obj['is_counted'] = bool(cos_scores[i] > similarity_threshold)
        final_count = sum(obj['is_counted'] for obj in st.session_state.objects)
    else:
        for obj in st.session_state.objects: obj['is_counted'] = False

    st.metric("Total Compté", final_count)

with col1:
    st.write(f"Détection : **{len(st.session_state.objects)}** candidats trouvés.")
    img_display = utils.overlay_objects(base_img, st.session_state.objects)
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click:
        if click_mode == " pipette couleur":
            # Calibrer et mettre à jour l'état des sliders
            new_hsv_params = utils.calibrate_hsv_from_click(img_hsv, click['x'], click['y'])
            st.session_state.update(new_hsv_params)
            st.rerun()
        
        elif click_mode == "Sélectionner un exemple" and st.session_state.objects:
            positions = np.array([(obj['cx'], obj['cy']) for obj in st.session_state.objects])
            idx = np.argmin(np.sum((positions - np.array([click["x"], click["y"]]))**2, axis=1))
            st.session_state.target_embedding = st.session_state.objects[idx]['embedding']
            st.rerun()

    with st.expander("Voir le masque de détection"):
        st.image(binary_mask, caption="Masque de détection en temps réel")
