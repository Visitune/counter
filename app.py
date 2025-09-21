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

st.set_page_config(page_title="Comptage Multi-Modal", layout="wide")
st.title("💡 Comptage Multi-Modal Interactif")

# --- Barre Latérale ---
with st.sidebar:
    st.header("1. Paramètres de Détection")
    st.info("Ajustez ces curseurs pour que les points gris apparaissent correctement sur chaque objet.")
    min_area = st.slider("Taille min objet (px²)", 10, 1000, 100)
    
    # NOUVEAUX CONTRÔLES CRUCIAUX
    block_size = st.slider("Taille du voisinage (impair)", 11, 101, 41, 2)
    C_value = st.slider("Sensibilité de détection (C)", -10, 10, 2)
    open_k = st.slider("Taille du nettoyage (px)", 1, 9, 3, 2)
    
    st.header("2. Paramètres de Comptage")
    selection_radius = st.slider("Rayon de clic (px)", 5, 50, 20)


# --- Chargement & Étape 1 : Détection et Embedding ---
st.header("Étape 1 : Charger une image & Ajuster la détection")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

# Charger le modèle d'IA
model = utils.get_clip_model()

@st.cache_data
def process_image(_img_bytes, _model, min_a, b_size, c_val, o_k):
    """La fonction de détection prend maintenant les paramètres de l'interface."""
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    candidates = utils.detect_and_embed_candidates(img, _model, min_a, b_size, c_val, o_k)
    return img, candidates

# La détection se relance automatiquement si vous bougez les curseurs de la sidebar
base_img, candidates = process_image(up.getvalue(), model, min_area, block_size, C_value, open_k)

if 'objects' not in st.session_state or st.session_state.get('image_name') != up.name:
    st.session_state.objects = candidates
    st.session_state.image_name = up.name
    st.session_state.target_embedding = None
# Mise à jour des objets si les paramètres changent
st.session_state.objects = candidates


# --- Interface Principale ---
col1, col2 = st.columns([3, 1])

with col2:
    st.header("Étape 2 : Guider l'IA")
    prompt_mode = st.radio("Méthode de guidage", ["Clic sur un exemple", "Description textuelle"])
    
    if prompt_mode == "Description textuelle":
        text_prompt = st.text_input("Que faut-il compter ?", "shrimp")
        if text_prompt:
            st.session_state.target_embedding = model.encode(text_prompt, convert_to_tensor=True)
        else:
            st.session_state.target_embedding = None

    st.header("Étape 3 : Ajuster & Compter")
    similarity_threshold = st.slider("Seuil de similarité", 0.5, 1.0, 0.85, 0.01)
    
    final_count = 0
    if 'target_embedding' in st.session_state and st.session_state.target_embedding is not None:
        if st.session_state.objects:
            all_embeddings = torch.stack([obj['embedding'] for obj in st.session_state.objects])
            cos_scores = util.cos_sim(st.session_state.target_embedding, all_embeddings)[0]
            
            for i, obj in enumerate(st.session_state.objects):
                if cos_scores[i] > similarity_threshold:
                    obj['is_counted'] = True
                    final_count += 1
                else:
                    obj['is_counted'] = False
    else:
        for obj in st.session_state.objects:
            obj['is_counted'] = False

    st.metric("Total Compté", final_count)


with col1:
    st.write(f"Détection initiale : **{len(st.session_state.objects)}** objets candidats trouvés.")
    img_display = utils.overlay_objects(base_img, st.session_state.objects)
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click and prompt_mode == "Clic sur un exemple":
        if st.session_state.objects:
            positions = np.array([(obj['cx'], obj['cy']) for obj in st.session_state.objects])
            distances_sq = np.sum((positions - np.array([click["x"], click["y"]]))**2, axis=1)
            idx = np.argmin(distances_sq)
            
            if np.sqrt(distances_sq[idx]) < selection_radius:
                st.session_state.target_embedding = st.session_state.objects[idx]['embedding']
                st.rerun()
