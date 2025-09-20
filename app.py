import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2, io, time
from math import sqrt

# === clics sur image (robuste Cloud) ===
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None  # on avertira plus bas


# ================= Utils =================
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points(image_pil, points, color=(0,255,0), radius=5):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def overlay_points_colored(image_pil, points, confirmed_ids=set(), rejected_ids=set(),
                           radius=6):
    """vert=confirmé, rouge=rejeté, gris=autres."""
    img = pil_to_cv(image_pil.copy())
    for i,(x,y) in enumerate(points):
        if i in confirmed_ids:  col = (0,255,0)
        elif i in rejected_ids: col = (0,0,255)
        else:                   col = (180,180,180)
        cv2.circle(img, (int(x),int(y)), radius, col, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def equalize_if_needed(gray, use_clahe, clip=3.0, tile=8):
    if not use_clahe: return gray
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)

def threshold_image(gray, method="otsu", invert=False, block_size=41, C=5):
    if method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY,
            block_size|1, C
        )
    else:
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

def connected_components(mask):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        comps.append(dict(idx=i, cx=float(cx), cy=float(cy), area=int(area)))
    return comps, labels, stats

def nearest_index(points, qx, qy):
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = (x-qx)**2 + (y-qy)**2
        if d < dmin:
            dmin, kmin = d, k
    return kmin, dmin**0.5


# === Segmentation guidée par confirmations/rejets (Lab + Mahalanobis) ===
def sample_disc_pixels(lab_img, cx, cy, r=8):
    H,W,_ = lab_img.shape
    x0, x1 = max(0, int(cx-r)), min(W, int(cx+r)+1)
    y0, y1 = max(0, int(cy-r)), min(H, int(cy+r)+1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
    return lab_img[y0:y1, x0:x1][mask]

def fit_gaussian(pixels):
    mu = pixels.mean(axis=0)
    cov = np.cov(pixels.T)
    cov = cov + 1e-6*np.eye(3)  # stabilisation
    inv = np.linalg.inv(cov)
    return mu, inv

def maha_batch(X, mu, inv):
    Z = X - mu
    return np.einsum('ij,jk,ik->i', Z, inv, Z, optimize=True)

def segment_from_hints(img_bgr, pos_pts, neg_pts, r=8, thr_factor=3.0, pos_bias=1.0,
                       open_k=3, close_k=3):
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    pos_pix = []
    for (x,y) in pos_pts:
        pos_pix.append(sample_disc_pixels(lab, x, y, r))
    pos_pix = np.concatenate(pos_pix) if pos_pix else np.empty((0,3), np.float32)

    neg_pix = []
    for (x,y) in neg_pts:
        neg_pix.append(sample_disc_pixels(lab, x, y, r))
    neg_pix = np.concatenate(neg_pix) if neg_pix else np.empty((0,3), np.float32)

    if len(pos_pix) < 30:
        return np.zeros((H,W), np.uint8)

    mu_pos, inv_pos = fit_gaussian(pos_pix)
    flat = lab.reshape(-1,3)
    dpos = maha_batch(flat, mu_pos, inv_pos)

    if len(neg_pix) >= 30:
        mu_neg, inv_neg = fit_gaussian(neg_pix)
        dneg = maha_batch(flat, mu_neg, inv_neg)
        pred = (dpos < pos_bias * dneg)
    else:
        dpos_pos = maha_batch(pos_pix, mu_pos, inv_pos)
        thr = thr_factor * np.median(dpos_pos)
        pred = (dpos < thr)

    mask = (pred.reshape(H,W) * 255).astype(np.uint8)
    mask = morph_cleanup(mask, open_k=open_k, close_k=close_k)
    return mask

# === Watershed pour séparer les objets collés ===
def split_touching_watershed(bw):
    # bw 0/255, objets = 255
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    sure_bg = cv2.dilate(bw, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img3c = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3c, markers)
    return markers  # int32

def centroids_from_markers(markers, min_area=80, max_area=50000):
    pts = []
    for lab in np.unique(markers):
        if lab <= 1:  # 0/1 = non-objets
            continue
        area = int((markers==lab).sum())
        if min_area <= area <= max_area:
            yx = np.argwhere(markers==lab).mean(axis=0)
            cy, cx = float(yx[0]), float(yx[1])
            pts.append((cx, cy))
    return pts


# ================= App =================
st.set_page_config(page_title="Comptage — validation guidée", layout="wide")
st.title("🧮 Comptage avec validation guidée (confirmer/rejeter → recomptage)")

with st.sidebar:
    st.header("Paramètres de base")
    produit = st.text_input("Produit (nom libre)", "objet")
    canal = st.selectbox("Canal initial", ["Gris (Y)", "HSV-S", "HSV-V", "Lab-a", "Lab-b"])
    use_clahe = st.checkbox("CLAHE (contraste local)", True)
    clahe_clip = st.slider("CLAHE clipLimit", 1.0, 6.0, 3.0, 0.1)
    clahe_tile = st.slider("CLAHE tileGrid", 4, 16, 8, 1)
    th_method = st.selectbox("Seuillage", ["Otsu", "Adaptive"])
    invert = st.checkbox("Inverser binaire (objets clairs)", True)
    block = st.slider("Adaptive: bloc (impair)", 15, 101, 41, 2)
    C = st.slider("Adaptive: C", -15, 15, 5, 1)
    open_k = st.slider("Ouverture (px)", 1, 15, 3, 1)
    close_k = st.slider("Fermeture (px)", 1, 15, 3, 1)
    min_area = st.slider("Aire min (px²)", 10, 20000, 120, 10)
    max_area = st.slider("Aire max (px²)", 100, 300000, 30000, 500)

    st.markdown("---")
    st.header("Validation guidée")
    rm_radius = st.slider("Rayon de sélection (px)", 5, 60, 20, 1)
    seed_radius = st.slider("Rayon échantillonnage couleur (px)", 4, 20, 8, 1)
    thr_factor = st.slider("Seuil (si pas de négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais pos/neg (si négatifs)", 0.5, 1.5, 1.0, 0.05)
    use_watershed = st.checkbox("Séparer objets collés (watershed)", True)

up = st.file_uploader("Déposer une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("1) Détection initiale → 2) Valider (confirmer 5–6 bons, rejeter 4–5 mauvais) → 3) Recompter → 4) Corrections fines → Export.")
    st.stop()

img = Image.open(up).convert("RGB")
st.image(img, use_column_width=True, caption="Image d’entrée")

if streamlit_image_coordinates is None:
    st.error("Le module `streamlit-image-coordinates` est manquant. "
             "Ajoute-le à `requirements.txt` puis redeploie : `streamlit-image-coordinates==0.1.5`.")
    st.stop()

# ===== 1) Détection initiale =====
if st.button("🚀 Détection initiale", type="primary"):
    t0 = time.time()
    img_cv = pil_to_cv(img)

    # Canal
    if canal.startswith("Gris"):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif canal == "HSV-S":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV); gray = hsv[:,:,1]
    elif canal == "HSV-V":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV); gray = hsv[:,:,2]
    elif canal == "Lab-a":
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB); gray = lab[:,:,1]
    else:
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB); gray = lab[:,:,2]

    gray = equalize_if_needed(gray, use_clahe, clahe_clip, clahe_tile)

    if th_method == "Adaptive":
        bw = threshold_image(gray, "adaptive", invert, block|1, C)
    else:
        bw = threshold_image(gray, "otsu", invert)

    bw = morph_cleanup(bw, open_k, close_k)

    comps, labels, stats = connected_components(bw)
    base_points = [(c["cx"], c["cy"]) for c in comps
                   if (min_area <= c["area"] <= max_area)]

    # état
    st.session_state["base_img"] = img
    st.session_state["bw_init"] = bw
    st.session_state["comps_init"] = comps
    st.session_state["points_init"] = base_points
    st.session_state["confirmed_ids"] = set()
    st.session_state["rejected_ids"]  = set()
    st.session_state["points_final"]  = base_points[:]  # sera mis à jour après recomptage

    st.success(f"Détection initiale: {len(base_points)} points • {(time.time()-t0)*1000:.0f} ms")
    st.image(overlay_points(img, base_points, (0,255,0), 6),
             use_column_width=True, caption="Overlay (détection initiale)")

# ===== 2) Validation guidée =====
if "points_init" in st.session_state:
    st.subheader("Validation guidée")
    mode = st.radio("Action sur clic", ["Confirmer un bon point", "Rejeter un point erroné"], horizontal=True)
    ov = overlay_points_colored(
        st.session_state["base_img"], st.session_state["points_init"],
        st.session_state["confirmed_ids"], st.session_state["rejected_ids"], radius=7
    )
    click = streamlit_image_coordinates(ov, key="val_click")

    if click:
        x, y = float(click["x"]), float(click["y"])
        idx, d = nearest_index(st.session_state["points_init"], x, y)
        if idx is not None and d <= rm_radius:
            if mode.startswith("Confirmer"):
                st.session_state["confirmed_ids"].add(idx)
                st.session_state["rejected_ids"].discard(idx)
            else:
                st.session_state["rejected_ids"].add(idx)
                st.session_state["confirmed_ids"].discard(idx)

    c1, c2, c3 = st.columns(3)
    c1.metric("Confirmés (positifs)", len(st.session_state["confirmed_ids"]))
    c2.metric("Rejetés (négatifs)", len(st.session_state["rejected_ids"]))
    c3.button("Réinitialiser sélection",
              on_click=lambda: (st.session_state["confirmed_ids"].clear(),
                                st.session_state["rejected_ids"].clear()))

    # ===== 3) Recompter avec indications =====
    if st.button("🎯 Recompter avec indications"):
        if len(st.session_state["confirmed_ids"]) < 3:
            st.warning("Confirme au moins 3–5 points pour orienter correctement la couleur.")
        else:
            t1 = time.time()
            pts_all = st.session_state["points_init"]
            pos_pts = [pts_all[i] for i in sorted(st.session_state["confirmed_ids"])]
            neg_pts = [pts_all[i] for i in sorted(st.session_state["rejected_ids"])]

            # Apprentissage couleur + segmentation
            mask_guided = segment_from_hints(
                pil_to_cv(st.session_state["base_img"]),
                pos_pts, neg_pts,
                r=seed_radius, thr_factor=thr_factor, pos_bias=pos_bias,
                open_k=open_k, close_k=close_k
            )

            # Adapter min/max area à partir des CONFIRMÉS (si dispo)
            areas_confirmed = []
            comps0 = st.session_state["comps_init"]
            for i in st.session_state["confirmed_ids"]:
                if i < len(comps0):
                    areas_confirmed.append(comps0[i]["area"])
            if len(areas_confirmed) >= 3:
                q10 = np.percentile(areas_confirmed, 10)
                q90 = np.percentile(areas_confirmed, 90)
                minA = max(10, int(0.7*q10))
                maxA = int(1.6*q90)
            else:
                minA, maxA = min_area, max_area

            # Séparation optionnelle des amas
            if use_watershed:
                markers = split_touching_watershed(mask_guided)
                pts = centroids_from_markers(markers, minA, maxA)
            else:
                compsG, _, _ = connected_components(mask_guided)
                pts = [(c["cx"], c["cy"]) for c in compsG if minA <= c["area"] <= maxA]

            st.session_state["points_final"] = pts
            st.success(f"Recomptage guidé: {len(pts)} points • {(time.time()-t1)*1000:.0f} ms")
            st.image(overlay_points(st.session_state["base_img"], pts, (0,255,0), 6),
                     use_column_width=True, caption="Overlay (recomptage guidé)")

# ===== 4) Corrections fines & Export =====
if "points_final" in st.session_state:
    st.subheader("Corrections fines (clic)")
    mode2 = st.radio("Action", ["Supprimer", "Ajouter"], horizontal=True, key="corrmode")
    ov2 = overlay_points(st.session_state["base_img"], st.session_state["points_final"], (0,255,0), 6)
    click2 = streamlit_image_coordinates(ov2, key="corr_click2")

    if click2:
        x, y = float(click2["x"]), float(click2["y"])
        pts = list(st.session_state["points_final"])
        if mode2 == "Supprimer":
            idx, d = nearest_index(pts, x, y)
            if idx is not None and d <= rm_radius and len(pts)>0:
                pts.pop(idx)
        else:
            idx, d = nearest_index(pts, x, y)
            if d > rm_radius:
                pts.append((x,y))
        st.session_state["points_final"] = pts

    st.metric("Compte final", len(st.session_state["points_final"]))
    st.image(overlay_points(st.session_state["base_img"], st.session_state["points_final"], (0,255,0), 6),
             use_column_width=True, caption="Overlay final")

    # Exports
    st.subheader("Export")
    buf = io.BytesIO()
    overlay_final = overlay_points(st.session_state["base_img"], st.session_state["points_final"], (0,255,0), 6)
    overlay_final.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"overlay_{produit}.png", mime="image/png")

    df = pd.DataFrame([{
        "produit": produit,
        "fichier_image": getattr(up, "name", "upload.png"),
        "nb_init": len(st.session_state.get("points_init", [])),
        "nb_confirmes": len(st.session_state.get("confirmed_ids", [])),
        "nb_rejetes": len(st.session_state.get("rejected_ids", [])),
        "nb_final": len(st.session_state["points_final"]),
        "canal": canal, "seuillage": th_method, "invert": bool(invert),
        "open_k": int(open_k), "close_k": int(close_k)
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"resume_{produit}.csv", mime="text/csv")
