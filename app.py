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
st.title("💡 Comptage Multi-Modal Inspiré par COUNTGD")

# --- Barre Latérale ---
with st.sidebar:
    st.header("1. Paramètres")
    min_area = st.slider("Taille min objet (px²)", 10, 1000, 100)
    selection_radius = st.slider("Rayon de clic (px)", 5, 50, 20)
    st.info("Le modèle d'IA est chargé une seule fois au démarrage.")

# --- Chargement & Étape 1 & 2 : Détection et Embedding ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

# Charger le modèle d'IA
model = utils.get_clip_model()

@st.cache_data
def process_image(_img_bytes, _model, min_a):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    candidates = utils.detect_and_embed_candidates(img, _model, min_a)
    return img, candidates

base_img, candidates = process_image(up.getvalue(), model, min_area)

# Initialisation de l'état de la session
if 'objects' not in st.session_state or st.session_state.get('image_name') != up.name:
    st.session_state.objects = candidates
    st.session_state.image_name = up.name
    # Très important : réinitialiser l'embedding cible si l'image change
    st.session_state.target_embedding = None

# --- Interface Principale ---
col1, col2 = st.columns([3, 1])

with col2:
    st.header("Étape 2 : Guider l'IA")
    prompt_mode = st.radio("Méthode de guidage", ["Clic sur un exemple", "Description textuelle"])

    # La logique de mise à jour de l'état est maintenant gérée directement ici
    if prompt_mode == "Description textuelle":
        text_prompt = st.text_input("Que faut-il compter ?", "shrimp")
        if text_prompt:
            # Met à jour l'état de la session directement
            st.session_state.target_embedding = model.encode(text_prompt, convert_to_tensor=True)
        else:
            # Si le texte est effacé, on réinitialise
            st.session_state.target_embedding = None

    st.header("Étape 3 : Ajuster & Compter")
    similarity_threshold = st.slider("Seuil de similarité", 0.5, 1.0, 0.85, 0.01)
    
    final_count = 0
    # La logique de comptage ne fait que LIRE l'état de la session
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
        # Si aucun cible n'est défini, s'assurer que rien n'est compté
        for obj in st.session_state.objects:
            obj['is_counted'] = False

    st.metric("Total Compté", final_count)
    st.info("Le comptage est mis à jour en temps réel lorsque vous bougez le curseur.")


with col1:
    img_display = utils.overlay_objects(base_img, st.session_state.objects)
    click = streamlit_image_coordinates(img_display, key="click_img")

    # La logique du clic met à jour l'état de la session directement
    if click and prompt_mode == "Clic sur un exemple":
        if st.session_state.objects:
            positions = np.array([(obj['cx'], obj['cy']) for obj in st.session_state.objects])
            distances_sq = np.sum((positions - np.array([click["x"], click["y"]]))**2, axis=1)
            idx = np.argmin(distances_sq)
            
            if np.sqrt(distances_sq[idx]) < selection_radius:
                st.session_state.target_embedding = st.session_state.objects[idx]['embedding']
                # On force le rechargement pour que la logique de comptage s'exécute
                st.rerun()
