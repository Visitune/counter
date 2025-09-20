# utils.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io

# ================= Conversion et Dessin =================

def pil_to_cv(img_pil):
    """Convertit une image PIL (RGB) en image OpenCV (BGR)."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    """Convertit une image OpenCV (BGR) en image PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points(image_pil, points, color=(0,255,0), radius=5):
    """Dessine des cercles sur une image PIL aux positions données."""
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def overlay_points_colored(image_pil, points, confirmed_ids=set(), rejected_ids=set(), radius=6):
    """Dessine des points colorés : vert (confirmé), rouge (rejeté), gris (autres)."""
    img = pil_to_cv(image_pil.copy())
    for i,(x,y) in enumerate(points):
        if i in confirmed_ids:  col = (0,255,0)
        elif i in rejected_ids: col = (0,0,255)
        else:                   col = (180,180,180)
        cv2.circle(img, (int(x),int(y)), radius, col, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)


# ================= Traitement d'Image Fondamental =================

def equalize_if_needed(gray, use_clahe, clip=3.0, tile=8):
    """Applique l'égalisation de contraste local (CLAHE) si activée."""
    if not use_clahe: return gray
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)

def threshold_image(gray, method="otsu", invert=False, block_size=41, C=5):
    """Binarise une image en niveaux de gris avec la méthode Otsu ou adaptative."""
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
    """Nettoie un masque binaire avec des opérations morphologiques."""
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def connected_components(mask):
    """Trouve les objets (composants connexes) dans un masque binaire."""
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        comps.append(dict(idx=i, cx=float(cx), cy=float(cy), area=int(area)))
    return comps, labels, stats

def nearest_index(points, qx, qy):
    """Trouve l'index du point le plus proche d'une coordonnée (qx, qy)."""
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = (x-qx)**2 + (y-qy)**2
        if d < dmin:
            dmin, kmin = d, k
    return kmin, dmin**0.5


# ================= Segmentation et Classification Couleur =================

def sample_disc_pixels(lab_img, cx, cy, r=8):
    """Échantillonne les pixels dans un disque de rayon r autour de (cx, cy)."""
    H,W,_ = lab_img.shape
    x0, x1 = max(0, int(cx-r)), min(W, int(cx+r)+1)
    y0, y1 = max(0, int(cy-r)), min(H, int(cy+r)+1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
    return lab_img[y0:y1, x0:x1][mask]

def fit_gaussian(pixels):
    """Calcule la moyenne et la matrice de covariance inverse pour un ensemble de pixels."""
    mu = pixels.mean(axis=0)
    cov = np.cov(pixels.T)
    cov = cov + 1e-6*np.eye(3)  # Stabilisation numérique
    inv_cov = np.linalg.inv(cov)
    return mu, inv_cov

def maha_batch(X, mu, inv_cov):
    """Calcule la distance de Mahalanobis pour un batch de points X."""
    Z = X - mu
    return np.einsum('ij,jk,ik->i', Z, inv_cov, Z, optimize=True)

def segment_from_hints(img_bgr, pos_pts, neg_pts, r=8, thr_factor=3.0, pos_bias=1.0, open_k=3, close_k=3):
    """SEGMENTATION LENTE (par pixel) : Crée un masque binaire à partir de points positifs/négatifs."""
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    pos_pix = np.concatenate([sample_disc_pixels(lab, x, y, r) for x,y in pos_pts]) if pos_pts else np.empty((0,3))
    neg_pix = np.concatenate([sample_disc_pixels(lab, x, y, r) for x,y in neg_pts]) if neg_pts else np.empty((0,3))
    
    if len(pos_pix) < 30: return np.zeros((H,W), np.uint8)

    mu_pos, inv_pos = fit_gaussian(pos_pix)
    flat_lab = lab.reshape(-1,3)
    d_pos = maha_batch(flat_lab, mu_pos, inv_pos)

    if len(neg_pix) >= 30:
        mu_neg, inv_neg = fit_gaussian(neg_pix)
        d_neg = maha_batch(flat_lab, mu_neg, inv_neg)
        pred = (d_pos < pos_bias * d_neg)
    else:
        d_pos_samples = maha_batch(pos_pix, mu_pos, inv_pos)
        thr = thr_factor * np.median(d_pos_samples)
        pred = (d_pos < thr)

    mask = (pred.reshape(H,W) * 255).astype(np.uint8)
    return morph_cleanup(mask, open_k, close_k)


# ================= Séparation d'objets (Watershed) =================

def split_touching_watershed(bw):
    """Sépare les objets connectés en utilisant l'algorithme Watershed."""
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
    return markers

def centroids_from_markers(markers, min_area=80, max_area=50000):
    """Extrait les centroïdes des régions identifiées par Watershed."""
    pts = []
    for lab in np.unique(markers):
        if lab <= 1: continue # 0/1 sont les marqueurs de fond/bordure
        area = int((markers==lab).sum())
        if min_area <= area <= max_area:
            yx = np.argwhere(markers==lab).mean(axis=0)
            pts.append((float(yx[1]), float(yx[0]))) # (cx, cy)
    return pts


# ================= Pipelines de Traitement Complets =================

@st.cache_data
def run_initial_detection(_img_bytes, canal, use_clahe, clahe_clip, clahe_tile,
                          th_method, invert, block, C, open_k, close_k,
                          min_area, max_area):
    """
    Exécute le pipeline complet de détection initiale.
    Mise en cache par Streamlit pour la performance.
    """
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    img_cv = pil_to_cv(img)

    # Sélection du canal de couleur
    if canal.startswith("Gris"): gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif canal == "HSV-S":       gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)[:,:,1]
    elif canal == "HSV-V":       gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)[:,:,2]
    elif canal == "Lab-a":       gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)[:,:,1]
    else:                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)[:,:,2]

    gray = equalize_if_needed(gray, use_clahe, clahe_clip, clahe_tile)
    
    # Binarisation et nettoyage
    if th_method == "Adaptive": bw = threshold_image(gray, "adaptive", invert, block, C)
    else:                       bw = threshold_image(gray, "otsu", invert)
    bw = morph_cleanup(bw, open_k, close_k)

    # Détection des objets
    comps, labels, _ = connected_components(bw)
    base_points = [(c["cx"], c["cy"]) for c in comps if min_area <= c["area"] <= max_area]

    return base_points, comps, labels, bw
