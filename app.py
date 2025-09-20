# app.py

import streamlit as st
from PIL import Image
import pandas as pd
import io

import utils

# Importation de la librairie pour les clics sur image
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Module `streamlit-image-coordinates` manquant. Veuillez l'ajouter à requirements.txt.")
    st.stop()

# ==================== Configuration de la Page ====================
st.set_page_config(page_title="Comptage Assisté par IA", layout="wide")
st.title("🧮 Comptage Assisté par Apprentissage Actif")

# ==================== Barre Latérale ====================
with st.sidebar:
    st.header("Paramètres")
    produit = st.text_input("Nom du produit", "objet")
    min_area = st.slider("Taille min objet (px²)", 10, 5000, 100)
    max_area = st.slider("Taille max objet (px²)", 1000, 100000, 30000)
    selection_radius = st.slider("Rayon de clic (px)", 5, 50, 20)

# ==================== Chargement & Détection Initiale (Étape 1) ====================
st.subheader("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image pour commencer", type=["jpg", "jpeg", "png"])

if not up:
    st.stop()

# Mise en cache de la détection initiale pour la performance
@st.cache_data
def get_initial_objects(_img_bytes, min_a, max_a):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    objects = utils.detect_candidate_objects(img, min_a, max_a)
    return img, objects

base_img, candidate_objects = get_initial_objects(up.getvalue(), min_area, max_area)

# Initialiser ou réinitialiser l'état de la session si l'image ou les paramètres changent
if "objects" not in st.session_state or st.session_state.get("image_name") != up.name:
    st.session_state.objects = candidate_objects
    st.session_state.image_name = up.name

# ==================== Interface Principale ====================
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Étape 2 : Guider l'IA")
    st.info("Cliquez sur l'image pour marquer quelques objets comme 'Bons' et d'autres comme 'Mauvais'.")
    mode_guidage = st.radio("Action du clic :", ["✅ Marquer comme Bon", "❌ Marquer comme Mauvais"], key="guidage")

    st.subheader("Étape 3 : Lancer l'IA")
    if st.button("🎯 Lancer l'apprentissage", type="primary", use_container_width=True):
        updated_objects, message = utils.train_and_predict(st.session_state.objects)
        st.session_state.objects = updated_objects
        if "Veuillez" in message:
            st.warning(message)
        else:
            st.success(message)
        st.rerun()

    st.subheader("Étape 4 : Corriger et Exporter")
    mode_correction = st.checkbox("Activer le mode correction")
    
    # Compteurs
    confirmed = sum(1 for o in st.session_state.objects if o['status'] == 'confirmed')
    rejected = sum(1 for o in st.session_state.objects if o['status'] == 'rejected')
    proposed_good = sum(1 for o in st.session_state.objects if o['status'] == 'proposed_good')
    final_count = confirmed + proposed_good

    st.metric("Total Final Estimé", final_count)
    st.write(f"Détails : {confirmed} manuels + {proposed_good} IA")
    
    # Export
    df = pd.DataFrame([{"produit": produit, "total_final": final_count}])
    st.download_button("⬇️ Exporter le résumé (CSV)", df.to_csv(index=False).encode('utf-8'),
                       f"resume_{produit}.csv", "text/csv", use_container_width=True)

with col1:
    img_display = utils.overlay_objects(base_img, st.session_state.objects)
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click:
        idx, dist = utils.nearest_object_index(st.session_state.objects, click["x"], click["y"])
        
        if idx is not None and dist < selection_radius:
            target_obj = st.session_state.objects[idx]
            
            # Logique de clic selon le mode
            if mode_correction:
                # En mode correction, on inverse les propositions de l'IA
                if target_obj['status'] == 'proposed_good':
                    target_obj['status'] = 'rejected'
                elif target_obj['status'] == 'proposed_bad':
                    target_obj['status'] = 'confirmed'
            else:
                # En mode guidage, on définit les exemples
                target_obj['status'] = "confirmed" if "Bon" in mode_guidage else "rejected"
            
            st.rerun()
