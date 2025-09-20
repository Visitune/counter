# app.py

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import io, time

# Importe toutes les fonctions de traitement depuis notre module utils
import utils

# === Configuration de la page et de l'état de session ===
st.set_page_config(page_title="Comptage Guidé", layout="wide")
if "init_done" not in st.session_state:
    st.session_state["init_done"] = False
    st.session_state["confirmed_ids"] = set()
    st.session_state["rejected_ids"] = set()
    st.session_state["points_final"] = []
    
# Importation de la librairie pour les clics sur image
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Le module `streamlit-image-coordinates` est manquant. "
             "Ajoutez-le à `requirements.txt` puis redéployez.")
    st.stop()


# ==================== INTERFACE UTILISATEUR ====================
st.title("🧮 Comptage avec validation guidée")
st.info("Workflow : 1) Détection initiale → 2) Guider (confirmer/rejeter des points) → 3) Recompter → 4) Corrections fines → 5) Export.")

# --- Barre latérale avec les paramètres ---
with st.sidebar:
    st.header("1. Paramètres de base")
    produit = st.text_input("Produit (nom libre)", "objet")
    canal = st.selectbox("Canal de couleur", ["Gris (Y)", "HSV-S", "HSV-V", "Lab-a", "Lab-b"])
    use_clahe = st.checkbox("Améliorer contraste (CLAHE)", True)
    th_method = st.selectbox("Méthode de seuillage", ["Otsu", "Adaptive"])
    invert = st.checkbox("Objets clairs sur fond sombre", True)
    open_k = st.slider("Nettoyage (ouverture)", 1, 15, 3, 1)
    close_k = st.slider("Bouchage trous (fermeture)", 1, 15, 3, 1)
    min_area = st.slider("Aire min (px²)", 10, 20000, 120, 10)
    max_area = st.slider("Aire max (px²)", 100, 300000, 30000, 500)
    
    st.markdown("---")
    st.header("2. Paramètres de guidage")
    rm_radius = st.slider("Rayon de sélection (clic)", 5, 60, 20, 1)
    seed_radius = st.slider("Rayon échantillonnage couleur", 4, 20, 8, 1)
    thr_factor = st.slider("Seuil couleur (si pas de rejets)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais Positifs/Négatifs", 0.5, 1.5, 1.0, 0.05)
    use_watershed = st.checkbox("Séparer objets collés (Watershed)", True)

# --- Chargement de l'image ---
up = st.file_uploader("Déposer une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.stop()

# On stocke l'image et ses octets en session pour éviter de les recharger
st.session_state["base_img"] = Image.open(up).convert("RGB")
st.session_state["img_bytes"] = up.getvalue()

st.image(st.session_state["base_img"], use_column_width=True, caption="Image d'entrée")

# ==================== ÉTAPE 1 : DÉTECTION INITIALE ====================

# Appelle la fonction de détection (qui est cachée) pour obtenir les résultats
# Streamlit ne la ré-exécutera que si les paramètres de la sidebar changent
points_init, comps_init, _, _ = utils.run_initial_detection(
    st.session_state["img_bytes"], canal, use_clahe, 3.0, 8, # Clahe clip/tile hardcodés ou à ajouter à la sidebar
    th_method, invert, 41, 5, # Adaptive block/C hardcodés ou à ajouter
    open_k, close_k, min_area, max_area
)

if st.button("🚀 1. Lancer la détection initiale", type="primary", use_container_width=True):
    st.session_state["points_init"] = points_init
    st.session_state["comps_init"] = comps_init
    st.session_state["confirmed_ids"] = set()
    st.session_state["rejected_ids"]  = set()
    st.session_state["points_final"]  = points_init[:]
    st.session_state["init_done"] = True

    st.success(f"Détection initiale trouvée : {len(points_init)} points.")
    st.image(utils.overlay_points(st.session_state["base_img"], points_init),
             use_column_width=True, caption="Résultat de la détection initiale")

# ==================== ÉTAPES 2 & 3 : VALIDATION ET RECOMPTAGE ====================

if st.session_state["init_done"]:
    st.subheader("2. Guidez l'algorithme par des clics")
    mode = st.radio("Action sur clic", ["✅ Confirmer (bon)", "❌ Rejeter (mauvais)"], horizontal=True, label_visibility="collapsed")
    
    # Affichage des points pour la validation
    img_for_validation = utils.overlay_points_colored(
        st.session_state["base_img"], st.session_state["points_init"],
        st.session_state["confirmed_ids"], st.session_state["rejected_ids"], radius=7
    )
    click = streamlit_image_coordinates(img_for_validation, key="val_click")

    if click:
        x, y = float(click["x"]), float(click["y"])
        idx, d = utils.nearest_index(st.session_state["points_init"], x, y)
        if idx is not None and d <= rm_radius:
            if mode.startswith("✅"):
                st.session_state["confirmed_ids"].add(idx)
                st.session_state["rejected_ids"].discard(idx)
            else:
                st.session_state["rejected_ids"].add(idx)
                st.session_state["confirmed_ids"].discard(idx)
            st.rerun() # Pour rafraîchir l'affichage immédiatement

    c1, c2, c3 = st.columns([1,1,2])
    c1.metric("Confirmés", len(st.session_state["confirmed_ids"]))
    c2.metric("Rejetés", len(st.session_state["rejected_ids"]))
    if c3.button("Réinitialiser la sélection"):
        st.session_state["confirmed_ids"].clear()
        st.session_state["rejected_ids"].clear()
        st.rerun()

    st.subheader("3. Mettre à jour le comptage")
    if st.button("🎯 Recompter avec les indications", use_container_width=True):
        if len(st.session_state["confirmed_ids"]) < 3:
            st.warning("Veuillez confirmer au moins 3 bons points pour un résultat optimal.")
        else:
            t1 = time.time()
            img_cv = utils.pil_to_cv(st.session_state["base_img"])
            
            # Segmentation guidée par couleur (maintenant beaucoup plus rapide)
            mask_guided = utils.segment_from_hints(
                img_cv,
                [st.session_state["points_init"][i] for i in st.session_state["confirmed_ids"]],
                [st.session_state["points_init"][i] for i in st.session_state["rejected_ids"]],
                r=seed_radius, thr_factor=thr_factor, pos_bias=pos_bias,
                open_k=open_k, close_k=close_k
            )

            # Option de séparation avec Watershed
            if use_watershed:
                markers = utils.split_touching_watershed(mask_guided)
                pts = utils.centroids_from_markers(markers, min_area, max_area)
            else:
                comps, _, _ = utils.connected_components(mask_guided)
                pts = [(c["cx"], c["cy"]) for c in comps if min_area <= c["area"] <= max_area]

            st.session_state["points_final"] = pts
            st.success(f"Recomptage guidé : {len(pts)} points trouvés en {(time.time()-t1)*1000:.0f} ms.")
    
    # Toujours afficher le dernier résultat du recomptage
    if st.session_state["points_final"]:
        st.image(utils.overlay_points(st.session_state["base_img"], st.session_state["points_final"]),
                 use_column_width=True, caption="Résultat après recomptage guidé")


# ==================== ÉTAPES 4 & 5 : CORRECTIONS FINES ET EXPORT ====================

if st.session_state["points_final"]:
    st.subheader("4. Corrections fines (Ajouter/Supprimer)")
    mode2 = st.radio("Action", ["🗑️ Supprimer", "➕ Ajouter"], horizontal=True, key="corrmode", label_visibility="collapsed")
    
    img_for_correction = utils.overlay_points(st.session_state["base_img"], st.session_state["points_final"])
    click2 = streamlit_image_coordinates(img_for_correction, key="corr_click")

    if click2:
        x, y = float(click2["x"]), float(click2["y"])
        pts = list(st.session_state["points_final"])
        
        if mode2 == "🗑️ Supprimer":
            idx, d = utils.nearest_index(pts, x, y)
            if idx is not None and d <= rm_radius and len(pts)>0:
                pts.pop(idx)
        else: # Ajouter
            # Ajoute seulement si le clic n'est pas trop près d'un point existant
            _, d = utils.nearest_index(pts, x, y)
            if d > rm_radius:
                pts.append((x,y))
        st.session_state["points_final"] = pts
        st.rerun()

    st.metric("Total final", len(st.session_state["points_final"]))
    st.image(utils.overlay_points(st.session_state["base_img"], st.session_state["points_final"]),
             use_column_width=True, caption="Comptage final après corrections")

    # --- Exports ---
    st.subheader("5. Exporter les résultats")
    c1, c2 = st.columns(2)
    
    # Export Image
    buf = io.BytesIO()
    overlay_final = utils.overlay_points(st.session_state["base_img"], st.session_state["points_final"])
    overlay_final.save(buf, format="PNG")
    c1.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"comptage_{produit}.png", mime="image/png", use_container_width=True)

    # Export CSV
    df = pd.DataFrame([{
        "produit": produit,
        "fichier": getattr(up, "name", "upload.png"),
        "total_final": len(st.session_state["points_final"]),
        "total_initial": len(st.session_state.get("points_init", [])),
        "nb_confirmes": len(st.session_state.get("confirmed_ids", set())),
        "nb_rejetes": len(st.session_state.get("rejected_ids", set())),
        "canal": canal, "seuillage": th_method
    }])
    c2.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"resume_{produit}.csv", mime="text/csv", use_container_width=True)
