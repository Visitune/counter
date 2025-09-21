# app.py

import streamlit as st
from PIL import Image
import utils
import io
import numpy as np

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Module `streamlit-image-coordinates` manquant.")
    st.stop()

st.set_page_config(page_title="Comptage par Clustering", layout="wide")
st.title("💡 Comptage Efficace par Clustering et Sélection")

# --- Barre Latérale ---
with st.sidebar:
    st.header("1. Paramètres de Détection")
    min_area = st.slider("Taille min objet (px²)", 10, 5000, 150)
    n_clusters = st.slider("Nombre de familles (clusters)", 2, 10, 5)

    st.header("2. Outil de Calibration Couleur")
    st.info("Utilisez la pipette pour pré-régler la couleur de l'objet d'intérêt.")
    # (Les sliders de couleur sont maintenant contrôlés par la pipette)

# --- Initialisation de l'état ---
if 'hsv_params' not in st.session_state:
    # Valeurs par défaut génériques
    st.session_state.hsv_params = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}
if 'selected_cluster_id' not in st.session_state:
    st.session_state.selected_cluster_id = None

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

model = utils.get_clip_model()

@st.cache_data
def load_image_data(_img_bytes):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    img_hsv = cv2.cvtColor(utils.pil_to_cv(img), cv2.COLOR_BGR2HSV)
    return img, img_hsv

base_img, img_hsv = load_image_data(up.getvalue())

# Logique de réinitialisation si l'image change
if st.session_state.get('img_name') != up.name:
    st.session_state.img_name = up.name
    st.session_state.selected_cluster_id = None
    # Réinitialiser les paramètres de couleur
    st.session_state.hsv_params = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}

# --- Détection, Embedding et Clustering ---
candidates = utils.detect_and_embed(base_img, model, st.session_state.hsv_params, min_area)
clustered_objects = utils.cluster_objects(candidates, n_clusters)

# --- Interface Principale ---
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Étape 2 : Sélectionner le groupe à compter")
    st.info("Cliquez sur un objet dans l'image pour sélectionner sa famille et lancer le comptage.")
    
    final_count = 0
    if st.session_state.selected_cluster_id is not None:
        final_count = sum(1 for obj in clustered_objects if obj['cluster_id'] == st.session_state.selected_cluster_id)
    
    st.metric("Total Compté", final_count)

    if st.button("Réinitialiser la sélection", use_container_width=True):
        st.session_state.selected_cluster_id = None
        st.rerun()

    with st.expander("Affiner la calibration couleur (avancé)"):
        hsv = st.session_state.hsv_params
        hsv['h_min'] = st.slider("H_min", 0, 179, hsv['h_min'])
        hsv['h_max'] = st.slider("H_max", 0, 179, hsv['h_max'])
        # ... Ajouter S et V si nécessaire

with col1:
    st.write(f"L'IA a identifié **{len(clustered_objects)}** objets potentiels et les a regroupés en **{n_clusters}** familles.")
    
    # Utiliser une image de base pour la pipette pour éviter de cliquer sur les points
    click = streamlit_image_coordinates(base_img, key="click_img")

    if click:
        # La pipette a la priorité pour calibrer la couleur
        new_hsv = utils.calibrate_hsv_from_click(img_hsv, click['x'], click['y'])
        st.session_state.hsv_params.update(new_hsv)
        
        # Trouver l'objet le plus proche du clic pour sélectionner le cluster
        if clustered_objects:
            positions = np.array([(obj['cx'], obj['cy']) for obj in clustered_objects])
            dist_sq = np.sum((positions - np.array([click['x'], click['y']]))**2, axis=1)
            if np.min(dist_sq) < (30**2): # Rayon de clic de 30px
                idx = np.argmin(dist_sq)
                st.session_state.selected_cluster_id = clustered_objects[idx]['cluster_id']
        st.rerun()

    # Afficher l'overlay après le traitement du clic
    img_display = utils.overlay_clustered_objects(base_img, clustered_objects, st.session_state.selected_cluster_id)
    st.image(img_display)
