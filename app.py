import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import io
from pathlib import Path

# -------------- Utils --------------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points(image_pil, points, color=(0,255,0), radius=5):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def equalize_if_needed(gray, use_clahe, clip=3.0, tile=8):
    if not use_clahe:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)

def threshold_image(gray, method="otsu", invert=False, block_size=31, C=5):
    if method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY,
            block_size|1, C
        )
    else:
        # Otsu
        _, th = cv2.threshold(
            gray, 0, 255,
            (cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
        )
    return th

def morph_cleanup(mask, open_k=3, close_k=3):
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def count_components(mask, min_area, max_area):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = cents[i]
            pts.append((float(cx), float(cy)))
    return pts

# -------------- App --------------
st.set_page_config(page_title="Comptage agrée — version de base (robuste Cloud)", layout="wide")
st.title("🧮 Comptage d’agréage — base robuste (OpenCV pur)")

with st.sidebar:
    st.header("Paramètres de traitement")
    produit = st.text_input("Produit (nom libre)", "objet")
    canal = st.selectbox("Canal de travail",
                         ["Gris (Y)", "HSV-S", "HSV-V", "Lab-a", "Lab-b"])
    use_clahe = st.checkbox("CLAHE (améliorer contraste local)", value=True)
    clahe_clip = st.slider("CLAHE clipLimit", 1.0, 6.0, 3.0, 0.1)
    clahe_tile = st.slider("CLAHE tileGrid", 4, 16, 8, 1)

    st.markdown("---")
    th_method = st.selectbox("Méthode de seuillage", ["Otsu", "Adaptive"])
    invert = st.checkbox("Inverser binaire (objets clairs)", value=True)
    block = st.slider("Adaptive: bloc (impair)", 15, 101, 41, 2)
    C = st.slider("Adaptive: C", -15, 15, 5, 1)

    st.markdown("---")
    open_k = st.slider("Ouverture (px)", 1, 15, 3, 1)
    close_k = st.slider("Fermeture (px)", 1, 15, 3, 1)
    min_area = st.slider("Aire min (px²)", 10, 20000, 120, 10)
    max_area = st.slider("Aire max (px²)", 100, 300000, 30000, 500)

    st.markdown("---")
    correction = st.number_input("Correction manuelle finale (+/-)", -9999, 9999, 0)

up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Dépose une image ci-dessus, ajuste les sliders → **Lancer le comptage**.")
    st.stop()

img = Image.open(up).convert("RGB")
st.image(img, use_column_width=True, caption="Image d’entrée")

if st.button("🚀 Lancer le comptage", type="primary"):
    img_cv = pil_to_cv(img)

    # Sélection du canal
    if canal.startswith("Gris"):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif canal == "HSV-S":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        gray = hsv[:, :, 1]
    elif canal == "HSV-V":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        gray = hsv[:, :, 2]
    elif canal == "Lab-a":
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 1]
    else:  # Lab-b
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 2]

    # Amélioration du contraste (utile en agro)
    gray = equalize_if_needed(gray, use_clahe, clip=clahe_clip, tile=clahe_tile)

    # Seuillage
    if th_method == "Adaptive":
        bw = threshold_image(gray, method="adaptive", invert=invert, block_size=block|1, C=C)
    else:
        bw = threshold_image(gray, method="otsu", invert=invert)

    # Nettoyage morphologique
    bw = morph_cleanup(bw, open_k=open_k, close_k=close_k)

    # Comptage
    points = count_components(bw, min_area=min_area, max_area=max_area)
    overlay = overlay_points(img, points, color=(0,255,0), radius=5)

    auto_count = len(points)
    final_count = max(0, auto_count + int(correction))

    st.metric("Compte automatique", auto_count)
    st.metric("Compte final (après correction)", final_count)

    c1, c2 = st.columns(2)
    with c1:
        st.image(overlay, use_column_width=True, caption="Overlay (points sur items)")
    with c2:
        st.image(bw, use_column_width=True, caption="Binaire utilisé (contrôle)")

    # Exports (image + CSV 1 ligne)
    st.subheader("Export")
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"overlay_{produit}.png", mime="image/png")

    df = pd.DataFrame([{
        "produit": produit,
        "fichier_image": getattr(up, "name", "upload.png"),
        "canal": canal,
        "method": th_method,
        "invert": bool(invert),
        "open_k": int(open_k),
        "close_k": int(close_k),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "compte_auto": int(auto_count),
        "correction": int(correction),
        "compte_final": int(final_count)
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"resume_{produit}.csv", mime="text/csv")
