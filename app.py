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
st.title("💡 Comptage par Clustering et Sélection")

# --- Initialisation de l'état ---
if 'hsv_params' not in st.session_state:
    st.session_state.hsv_params = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}

# --- Barre Latérale ---
with st.sidebar:
    st.header("1. Paramètres de Détection")
    min_area = st.slider("Taille min objet (px²)", 10, 5000, 150)
    n_clusters = st.slider("Nombre de familles (clusters)", 2, 10, 5)

    st.header("2. Outil de Calibration Couleur")
    st.info("Utilisez la pipette sur l'image pour régler la couleur de l'objet d'intérêt.")
    with st.expander("Affiner la calibration (avancé)"):
        hsv = st.session_state.hsv_params
        hsv['h_min'] = st.slider("H_min", 0, 179, hsv['h_min'])
        hsv['h_max'] = st.slider("H_max", 0, 179, hsv['h_max'])
        hsv['s_min'] = st.slider("S_min", 0, 255, hsv['s_min'])
        hsv['s_max'] = st.slider("S_max", 0, 255, hsv['s_max'])
        hsv['v_min'] = st.slider("V_min", 0, 255, hsv['v_min'])
        hsv['v_max'] = st.slider("V_max", 0, 255, hsv['v_max'])

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])

if not up: st.stop()

model = utils.get_clip_model()

# --- GESTION DE L'ÉTAT ET DU CACHE ---
# Si l'image change, on réinitialise tout
if st.session_state.get('img_name') != up.name:
    st.session_state.img_name = up.name
    st.session_state.objects = None
    st.session_state.selected_cluster_id = None
    st.session_state.hsv_params = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}

@st.cache_data
def get_base_images(_img_bytes):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    img_hsv = utils.pil_to_cv(img)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
    return img, img_hsv

base_img, img_hsv = get_base_images(up.getvalue())

# LA DÉTECTION N'EST PLUS CACHÉE, ELLE DÉPEND DES SLIDERS
# Mais on ne la stocke qu'une fois par cycle
objects, binary_mask, _ = utils.process_image(base_img, model, st.session_state.hsv_params, min_area, n_clusters)
st.session_state.objects = objects

# --- Interface Principale ---
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Étape 2 : Sélectionner le groupe à compter")
    st.info("Cliquez sur un objet dans l'image pour sélectionner sa famille. Utilisez la pipette pour calibrer la couleur.")
    
    final_count = 0
    if 'selected_cluster_id' in st.session_state and st.session_state.selected_cluster_id is not None:
        final_count = sum(1 for obj in st.session_state.objects if obj['cluster_id'] == st.session_state.selected_cluster_id)
    
    st.metric("Total Compté", final_count)

    if st.button("Réinitialiser la sélection", use_container_width=True):
        st.session_state.selected_cluster_id = None
        st.rerun()

with col1:
    st.write(f"L'IA a identifié **{len(st.session_state.objects)}** objets potentiels et les a regroupés en **{n_clusters}** familles.")
    
    # LA LOGIQUE DU CLIC EST MAINTENANT SÉPARÉE ET CLAIRE
    click_mode = st.radio("Action du clic :", ["Pipette Couleur", "Sélectionner une famille"], horizontal=True, label_visibility="collapsed")
    
    img_display = utils.overlay_clustered_objects(base_img, st.session_state.objects, st.session_state.get('selected_cluster_id'))
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click:
        if click_mode == "Pipette Couleur":
            new_hsv = utils.calibrate_hsv_from_click(img_hsv, click['x'], click['y'])
            if new_hsv:
                st.session_state.hsv_params.update(new_hsv)
                st.session_state.selected_cluster_id = None # Réinitialiser la sélection car les objets vont changer
                st.rerun()
        
        elif click_mode == "Sélectionner une famille":
            if st.session_state.objects:
                positions = np.array([(obj['cx'], obj['cy']) for obj in st.session_state.objects])
                dist_sq = np.sum((positions - np.array([click['x'], click['y']]))**2, axis=1)
                
                if dist_sq.any() and np.min(dist_sq) < (30**2):
                    idx = np.argmin(dist_sq)
                    st.session_state.selected_cluster_id = st.session_state.objects[idx]['cluster_id']
                    st.rerun()

    with st.expander("Voir le masque de détection"):
        st.image(binary_mask)
