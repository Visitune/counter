import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json, time

from utils import draw_rois, pil_to_cv, cv_to_pil, apply_rois_mask, adaptive_preprocess, morph_cleanup, count_connected_components, overlay_points, make_heatmap_from_points

st.set_page_config(page_title="Comptage agrée — Basic (OpenCV)", layout="wide")

DATA_DIR = Path("data")
JOURNAL_DIR = DATA_DIR / "journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_CSV = JOURNAL_DIR / "journal_comptage_agreage.csv"

def init_state():
    if "journal_rows" not in st.session_state:
        st.session_state.journal_rows = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
init_state()

# ===== Sidebar: Contexte & Paramètres =====
with st.sidebar:
    st.header("Contexte & Paramètres")
    colA, colB = st.columns(2)
    with colA:
        site = st.text_input("Site")
        ligne = st.text_input("Ligne")
        poste = st.text_input("Poste")
        operateur = st.text_input("Opérateur")
    with colB:
        produit = st.selectbox("Produit", ["Crevettes cuites","Riz long A","Lentilles vertes","Autre"])
        lot = st.text_input("Lot")
        fournisseur = st.text_input("Fournisseur")
        format_unite = st.text_input("Format unité (ex. pcs/kg)")

    st.markdown("---")
    prompt = st.text_input("Prompt texte (pour le log)", "shrimp tail-on cooked")

    st.markdown("**Paramètres traitement**")
    min_area = st.slider("Aire min composant (px)", 5, 200, 25, 5)
    max_area = st.slider("Aire max composant (px)", 500, 20000, 5000, 100)
    open_ksize = st.slider("Morph. ouverture (px)", 1, 11, 3, 1)
    close_ksize = st.slider("Morph. fermeture (px)", 1, 11, 3, 1)

    st.markdown("---")
    seuil_mae = st.number_input("Seuil MAE (tolérance)", 0, 50, 2)

st.title("🧮 Comptage d’agréage — Basic (OpenCV)")

tab1, tab2, tab3 = st.tabs(["Comptage", "Contrôle qualité", "Journal & Export"])

# ===== Onglet 1: Comptage =====
with tab1:
    up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
    else:
        img = None

    if img is not None:
        st.subheader("Image brute")
        st.image(img, use_container_width=True)

        rois = []
        st.markdown("**Dessine (optionnel) 2–5 boîtes exemplaires / zones à compter.**")
        canvas = st_canvas(
            fill_color="rgba(255,0,0,0.0)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="rect",
            key="canvas",
        )
        if canvas.json_data and "objects" in canvas.json_data:
            for o in canvas.json_data["objects"]:
                x, y, w, h = o["left"], o["top"], o["width"], o["height"]
                if w > 2 and h > 2:
                    rois.append([int(x), int(y), int(x+w), int(y+h)])
        if rois:
            st.caption(f"{len(rois)} zone(s) dessinée(s).")
            st.image(draw_rois(img, rois), caption="Zones d'intérêt", use_container_width=True)
        else:
            st.info("Aucune zone dessinée — l’algorithme comptera sur toute l’image.")

        if st.button("🚀 Lancer le comptage", type="primary"):
            t0 = time.time()
            img_cv = pil_to_cv(img)
            img_cv_proc = apply_rois_mask(img_cv, rois)

            bw = adaptive_preprocess(img_cv_proc, invert=True)
            bw = morph_cleanup(bw, open_ksize=open_ksize, close_ksize=close_ksize)

            points = count_connected_components(bw, min_area=min_area, max_area=max_area)
            count = len(points)
            overlay = overlay_points(img, points, color=(0,255,0), radius=4)
            heatmap = make_heatmap_from_points(img, points, sigma=12)

            dt_ms = (time.time() - t0) * 1000
            st.metric("Compte estimé", count)
            st.caption(f"Temps de traitement ~ {dt_ms:.0f} ms")

            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay, caption="Overlay détections", use_container_width=True)
            with col2:
                st.image(heatmap, caption="Heatmap", use_container_width=True)

            st.session_state.last_result = dict(
                fichier_image=getattr(up, "name", "upload.png"),
                compte=count, rois=rois, prompt=prompt,
                overlay=overlay, heatmap=heatmap, raw=img
            )

# ===== Onglet 2: Contrôle qualité =====
with tab2:
    if st.session_state.last_result is None:
        st.info("Réalise d’abord un comptage dans l’onglet **Comptage**.")
    else:
        lr = st.session_state.last_result
        st.subheader("Comparaison avec comptage manuel")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("Compte estimé")
            st.metric(label="", value=lr["compte"])
        with c2:
            compte_verif = st.number_input("Compte vérifié (manuel)", 0, 1_000_000, 0)
        with c3:
            ecart_abs = abs(lr["compte"] - compte_verif)
            st.write("Écart absolu")
            st.metric(label="", value=ecart_abs)

        ok = ecart_abs <= seuil_mae
        st.success("✅ Validation OK") if ok else st.error("❌ Validation NOK")

        obs = st.text_area("Observations")
        actions = st.text_area("Actions correctives (si NOK)")

        if st.button("📝 Ajouter au journal CSV"):
            now_iso = pd.Timestamp.now().isoformat()
            row = dict(
                site=site, ligne=ligne, poste=poste, date_heure=now_iso, opérateur=operateur,
                produit=produit, lot=lot, fournisseur=fournisseur, format_unité=format_unite,
                prompt_texte=lr["prompt"], nb_exemplaires_annotés=len(lr["rois"]),
                coords_exemplaires_xyxy=json.dumps(lr["rois"]),
                fichier_image=lr["fichier_image"],
                compte_prédit=lr["compte"], compte_vérifié=compte_verif,
                écart_abs=ecart_abs, seuil_acceptation_MAE=seuil_mae,
                validation_OK=bool(ok), observations=obs, actions_correctives=actions
            )
            st.session_state.journal_rows.append(row)

            base = Path(lr["fichier_image"]).stem
            overlay_path = JOURNAL_DIR / f"{base}_overlay.png"
            heatmap_path = JOURNAL_DIR / f"{base}_heatmap.png"
            raw_path = JOURNAL_DIR / f"{base}_raw.png"
            lr["overlay"].save(overlay_path)
            lr["heatmap"].save(heatmap_path)
            lr["raw"].save(raw_path)
            st.success(f"Ajouté au journal. Images: {overlay_path.name}, {heatmap_path.name}")

# ===== Onglet 3: Journal & Export =====
with tab3:
    df = pd.DataFrame(st.session_state.journal_rows)
    st.subheader("Journal courant (session)")
    st.dataframe(df, use_container_width=True, height=350)

    if st.button("💾 Sauvegarder/Mettre à jour CSV global"):
        header = not JOURNAL_CSV.exists() or JOURNAL_CSV.stat().st_size == 0
        df.to_csv(JOURNAL_CSV, mode="a", header=header, index=False, encoding="utf-8")
        st.success(f"Mise à jour CSV: {JOURNAL_CSV}")

    if JOURNAL_CSV.exists():
        with open(JOURNAL_CSV, "rb") as f:
            st.download_button("⬇️ Télécharger le journal CSV global", f, file_name=JOURNAL_CSV.name)
