import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json, time, cv2

# ===== Helpers (garde tout dans ce fichier pour simplicité Streamlit Cloud) =====
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def draw_rois(image_pil, rois, color=(0,255,0), width=3):
    img = image_pil.copy()
    d = ImageDraw.Draw(img)
    for x1,y1,x2,y2 in rois:
        d.rectangle([x1,y1,x2,y2], outline=(color[0],color[1],color[2]), width=width)
    return img

def overlay_points(image_pil, points, color=(0,255,0), radius=4):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def make_heatmap_from_points(image_pil, points, sigma=12, alpha_img=0.6, alpha_hm=0.6):
    h, w = image_pil.height, image_pil.width
    hm = np.zeros((h, w), dtype=np.float32)
    for (x, y) in points:
        if 0 <= x < w and 0 <= y < h:
            hm[int(y), int(x)] += 1.0
    hm = cv2.GaussianBlur(hm, (0,0), sigma)
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(pil_to_cv(image_pil), alpha_img, hm_color, alpha_hm, 0)
    return cv_to_pil(overlay)

def apply_rois_mask(img_cv, rois):
    if not rois:
        return img_cv
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for x1,y1,x2,y2 in rois:
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    return cv2.bitwise_and(img_cv, img_cv, mask=mask)

# --- MODE A : compter les QUEUES par couleur (HSV) ---
def count_tails_hsv(img_cv, rois, hmin, hmax, smin, vmin,
                    open_ksize, close_ksize, min_area, max_area):
    work = apply_rois_mask(img_cv, rois)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    # support plage de teinte wrap-around (ex : hmin>hmax)
    if hmin <= hmax:
        mask = cv2.inRange(hsv, (hmin, smin, vmin), (hmax, 255, 255))
    else:
        m1 = cv2.inRange(hsv, (0, smin, vmin), (hmax, 255, 255))
        m2 = cv2.inRange(hsv, (hmin, smin, vmin), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
    return points, mask

# --- MODE B : compter les CORPS par watershed (distance transform) ---
def count_bodies_watershed(img_cv, rois, clip_limit, tile_grid, invert_bin,
                           open_ksize, close_ksize, dt_ratio, min_area, max_area):
    work = apply_rois_mask(img_cv, rois)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    # CLAHE pour limiter l’influence des reflets/ombres
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    gray = clahe.apply(gray)

    # Otsu binaire (inversion optionnelle selon contraste)
    flag = cv2.THRESH_BINARY_INV if invert_bin else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)

    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    # sure background / sure foreground
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(bw, kernel, iterations=2)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dt_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # marqueurs (watershed)
    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    ws_input = work.copy()
    cv2.watershed(ws_input, markers)

    # chaque label > 1 est un objet ; -1 = frontières
    points = []
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        mask = np.uint8(markers == lbl) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            points.append((cx, cy))
    return points, bw

# ===== App =====
st.set_page_config(page_title="Comptage agréage — Basic (OpenCV)", layout="wide")

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
    prompt = st.text_input("Prompt (log)", "shrimp tail-on cooked")

    # Choix du MODE de comptage
    mode = st.selectbox("Mode de comptage", ["Queues (HSV)", "Corps (Watershed)"])

    if mode == "Queues (HSV)":
        st.caption("Cible: queues orange. Robuste quand les crevettes se chevauchent.")
        hmin = st.slider("Hue min (0–179)", 0, 179, 5)
        hmax = st.slider("Hue max (0–179)", 0, 179, 25)
        smin = st.slider("S min (0–255)", 0, 255, 80)
        vmin = st.slider("V min (0–255)", 0, 255, 60)
        open_ksize = st.slider("Ouverture (px)", 1, 15, 3)
        close_ksize = st.slider("Fermeture (px)", 1, 15, 3)
        min_area = st.slider("Aire min (px²)", 5, 2000, 50, 5)
        max_area = st.slider("Aire max (px²)", 100, 20000, 3000, 50)
    else:
        st.caption("Cible: corps entiers avec séparation des contacts (watershed).")
        clip_limit = st.slider("CLAHE clipLimit", 1.0, 6.0, 3.0, 0.1)
        tile_grid = st.slider("CLAHE tileGrid", 4, 16, 8, 1)
        invert_bin = st.checkbox("Inverser binaire (objets clairs)", value=True)
        open_ksize = st.slider("Ouverture (px)", 1, 15, 3)
        close_ksize = st.slider("Fermeture (px)", 1, 15, 3)
        dt_ratio = st.slider("Seuil distanceTransform (ratio)", 0.20, 0.80, 0.45, 0.01)
        min_area = st.slider("Aire min (px²)", 50, 30000, 800, 50)
        max_area = st.slider("Aire max (px²)", 1000, 120000, 30000, 500)

    st.markdown("---")
    seuil_mae = st.number_input("Seuil MAE (tolérance)", 0, 50, 2)

st.title("🧮 Comptage d’agréage — Basic (OpenCV orienté crevettes)")

tab1, tab2, tab3 = st.tabs(["Comptage", "Contrôle qualité", "Journal & Export"])

# ===== Onglet 1: Comptage =====
with tab1:
    up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
    img = Image.open(up).convert("RGB") if up else None

    if img is not None:
        st.subheader("Image brute")
        st.image(img, use_column_width=True)

        # ROIs (option)
        rois = []
        st.markdown("**Dessine (optionnel) des zones à compter (ROIs).**")
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
        else:
            st.info("Aucune zone dessinée — comptage sur toute l’image.")

        if st.button("🚀 Lancer le comptage", type="primary"):
            t0 = time.time()
            img_cv = pil_to_cv(img)

            if mode == "Queues (HSV)":
                points, debug_mask = count_tails_hsv(
                    img_cv, rois, hmin, hmax, smin, vmin,
                    open_ksize, close_ksize, min_area, max_area
                )
                debug_view = cv_to_pil(cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR))
            else:
                points, debug_mask = count_bodies_watershed(
                    img_cv, rois, clip_limit, tile_grid, invert_bin,
                    open_ksize, close_ksize, dt_ratio, min_area, max_area
                )
                debug_view = cv_to_pil(cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR))

            count = len(points)
            overlay = overlay_points(img, points, color=(0,255,0), radius=4)
            heatmap = make_heatmap_from_points(img, points, sigma=12)
            dt_ms = (time.time() - t0) * 1000

            st.metric("Compte estimé", count)
            st.caption(f"Temps de traitement ~ {dt_ms:.0f} ms  •  Mode: {mode}")

            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay, caption="Overlay détections", use_column_width=True)
            with col2:
                st.image(heatmap, caption="Heatmap", use_column_width=True)
            st.expander("Voir masque/debug").image(debug_view, use_column_width=True)

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
