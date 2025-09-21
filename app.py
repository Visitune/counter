# app.py

import streamlit as st
from PIL import Image
import utils
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Module `streamlit-image-coordinates` manquant.")
    st.stop()

st.set_page_config(page_title="Comptage par l'Exemple", layout="wide")
st.title("🎯 Comptage par l'Exemple")

# --- Initialisation de l'état ---
if 'positive_examples' not in st.session_state:
    st.session_state.positive_examples = []

# --- Barre Latérale ---
with st.sidebar:
    st.header("Paramètres")
    min_area = st.slider("Sensibilité de détection (px²)", 10, 1000, 150)
    similarity_threshold = st.slider("Seuil de similarité", 0.50, 1.00, 0.90, 0.01)

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

model = utils.get_clip_model()

# --- DÉTECTION ET CACHE ---
# Cette fonction ne s'exécute qu'une fois par image et par paramètre min_area
@st.cache_data
def run_detection(_img_bytes, _model, _min_area):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    objects = utils.detect_and_embed_candidates(img, _model, _min_area)
    return img, objects

base_img, all_objects = run_detection(up.getvalue(), model, min_area)

# Réinitialiser si l'image change
if st.session_state.get('img_name') != up.name:
    st.session_state.img_name = up.name
    st.session_state.positive_examples = []

# --- Interface Principale ---
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Étape 2 : Guider l'IA")
    st.info("Cliquez sur 2 ou 3 objets que vous souhaitez compter. Chaque clic ajoute un exemple.")
    
    num_examples = len(st.session_state.positive_examples)
    st.write(f"**{num_examples}** exemple(s) sélectionné(s).")
    
    if st.button("Réinitialiser les exemples", use_container_width=True):
        st.session_state.positive_examples = []
        st.rerun()

    # --- LOGIQUE DE COMPTAGE ---
    final_count = 0
    if num_examples > 0:
        # Créer l'empreinte moyenne des exemples
        example_embeddings = np.array([all_objects[i]['embedding'] for i in st.session_state.positive_examples])
        target_embedding = np.mean(example_embeddings, axis=0).reshape(1, -1)
        
        # Comparer à tous les autres objets
        all_embeddings = np.array([obj['embedding'] for obj in all_objects])
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]
        
        for i, obj in enumerate(all_objects):
            if similarities[i] > similarity_threshold:
                obj['is_counted'] = True
            else:
                obj['is_counted'] = False
        final_count = sum(obj.get('is_counted', False) for obj in all_objects)
    else:
        # Si pas d'exemple, rien n'est compté
        for obj in all_objects:
            obj['is_counted'] = False

    st.header("Étape 3 : Résultat")
    st.metric("Total Compté", final_count)

with col1:
    st.write(f"Détection initiale : **{len(all_objects)}** objets potentiels trouvés.")
    
    img_display = utils.overlay_objects(base_img, all_objects, st.session_state.positive_examples)
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click:
        if all_objects:
            positions = np.array([(obj['cx'], obj['cy']) for obj in all_objects])
            dist_sq = np.sum((positions - np.array([click['x'], click['y']]))**2, axis=1)
            
            # Ajouter le point cliqué comme exemple s'il est assez proche
            if np.min(dist_sq) < (30**2):
                idx = np.argmin(dist_sq)
                if idx not in st.session_state.positive_examples:
                    st.session_state.positive_examples.append(idx)
                    st.rerun()
