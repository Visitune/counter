# app.py

import streamlit as st
from PIL import Image
import pandas as pd
import io

# Importe les nouvelles fonctions utilitaires
import utils

# Importation de la librairie pour les clics sur image
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Le module `streamlit-image-coordinates` est manquant. "
             "Ajoutez-le à `requirements.txt` puis redéployez.")
    st.stop()

# ==================== Configuration de la Page ====================
st.set_page_config(page_title="Comptage Interactif", layout="wide")
st.title("🧮 Comptage par Classification Interactive")
st.info("Cliquez sur les objets pour les marquer comme 'Bons' (✅) ou 'Mauvais' (❌). L'IA apprendra et proposera des classifications en temps réel.")

# ==================== Barre Latérale ====================
with st.sidebar:
    st.header("Paramètres")
    produit = st.text_input("Nom du produit", "objet")
    min_area = st.slider("Taille minimale d'objet (px²)", 10, 5000, 100)
    max_area = st.slider("Taille maximale d'objet (px²)", 1000, 100000, 30000)
    selection_radius = st.slider("Rayon de clic (px)", 5, 50, 20)

# ==================== Chargement de l'Image ====================
up = st.file_uploader("1. Déposez une image ici", type=["jpg", "jpeg", "png"])

if not up:
    st.stop()

# Utiliser le cache de Streamlit pour ne lancer la détection qu'une seule fois par image
@st.cache_data
def get_candidate_objects(_img_bytes, min_a, max_a):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    objects = utils.detect_candidate_objects(img, min_a, max_a)
    return img, objects

# Exécuter la détection initiale
base_img, candidate_objects = get_candidate_objects(up.getvalue(), min_area, max_area)

# Initialiser l'état de la session si les objets changent
if "objects" not in st.session_state or st.session_state.get("image_name") != up.name:
    st.session_state.objects = candidate_objects
    st.session_state.image_name = up.name

# ==================== Interface Principale ====================
st.subheader("2. Guidez l'IA en cliquant")

# Créer deux colonnes pour l'image et les contrôles
col1, col2 = st.columns([3, 1])

with col2:
    mode = st.radio("Action du clic :", ["✅ Confirmer", "❌ Rejeter"], horizontal=True)
    
    if st.button("Réinitialiser toutes les étiquettes", use_container_width=True):
        for obj in st.session_state.objects:
            obj['status'] = 'neutral'
        st.rerun()

    # Compteurs
    confirmed_count = sum(1 for obj in st.session_state.objects if obj['status'] == 'confirmed')
    rejected_count = sum(1 for obj in st.session_state.objects if obj['status'] == 'rejected')
    proposed_good_count = sum(1 for obj in st.session_state.objects if obj['status'] == 'proposed_good')
    final_count = confirmed_count + proposed_good_count
    
    st.metric("✅ Confirmés manuellement", confirmed_count)
    st.metric("❌ Rejetés manuellement", rejected_count)
    st.metric("🤖 Proposés par l'IA", proposed_good_count)
    st.metric("📊 Total Final Provisoire", final_count, delta_color="off")
    
    st.subheader("3. Export")
    df = pd.DataFrame({
        "produit": [produit],
        "total": [final_count]
    })
    st.download_button("⬇️ Résumé (CSV)", df.to_csv(index=False).encode('utf-8'), 
                       f"resume_{produit}.csv", "text/csv", use_container_width=True)

with col1:
    # Dessiner l'overlay interactif
    img_display = utils.overlay_objects(base_img, st.session_state.objects)
    click = streamlit_image_coordinates(img_display, key="click_img")

    if click:
        x, y = click["x"], click["y"]
        idx, dist = utils.nearest_object_index(st.session_state.objects, x, y)
        
        if idx is not None and dist < selection_radius:
            # Appliquer la nouvelle étiquette
            new_status = "confirmed" if mode.startswith("✅") else "rejected"
            st.session_state.objects[idx]['status'] = new_status
            
            # Mettre à jour les classifications en temps réel
            st.session_state.objects = utils.update_classifications(st.session_state.objects)
            
            # Rafraîchir l'interface
            st.rerun()
