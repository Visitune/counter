import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import io, json, time, cv2, base64
from math import sqrt

# ============== Utilitaires simples ==============
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points_pil(image_pil, points, color=(0,255,0), radius=5):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def mask_from_strokes(canvas_json, h, w):
    """Convertit les traits du canvas (libre) en masque binaire h×w."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not canvas_json or "objects" not in canvas_json:
        return mask
    for obj in canvas_json["objects"]:
        if obj.get("type") == "path":
            # convertir la polyline du trait en polygone approx. et la rasteriser
            path = obj.get("path", [])
            if not path: 
                continue
            pts = []
            for seg in path:
                if len(seg) >= 3:
                    _, x, y = seg[:3]
                    pts.append((int(x), int(y)))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.array(pts, dtype=np.int32)], False, 255, thickness=max(1, int(obj.get("strokeWidth",2))))
    return mask

def nearest_index(points, qx, qy):
    """Renvoie l'indice du point le plus proche de (qx,qy)."""
    if not points:
        return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = sqrt((x-qx)**2 + (y-qy)**2)
        if d < dmin:
            dmin, kmin = d, k
    return kmin, dmin

# ============== Segmentation guidée par exemples ==============
def segment_from_seeds(img_bgr, pos_mask, neg_mask, thr_factor=3.0, pos_bias=1.0,
                       open_ksize=3, close_ksize=3):
    """
    img_bgr : image BGR (OpenCV)
    pos_mask / neg_mask : masques uint8 (255=traits)
    Sortie: mask_items uint8 {0,255}
    """
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)

    pos_idx = np.where(pos_mask.reshape(-1) > 0)[0]
    neg_idx = np.where(neg_mask.reshape(-1) > 0)[0]

    if len(pos_idx) < 10:
        return np.zeros((H,W), np.uint8)  # pas assez d'info

    mu_pos = lab[pos_idx].mean(axis=0)
    if len(neg_idx) >= 10:
        mu_neg = lab[neg_idx].mean(axis=0)
        # classification par distance euclidienne aux barycentres
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        dneg = np.linalg.norm(lab - mu_neg, axis=1)
        # pos si plus proche de pos *et* dpos < pos_bias*dneg
        pred = (dpos < (pos_bias * dneg)).astype(np.uint8)
    else:
        # pas de négatif: seuillage à partir de la distance au centre positif
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        # seuil = median(dist des pixels positifs) * thr_factor
        ref = np.median(dpos[pos_idx]) if len(pos_idx) else dpos.mean()
        pred = (dpos < (thr_factor * ref)).astype(np.uint8)

    mask = (pred.reshape(H,W) * 255).astype(np.uint8)

    # Nettoyage morpho
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask

def count_from_mask(mask, min_area=80, max_area=50000):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
    return points

# ============== App Streamlit ==============
st.set_page_config(page_title="Comptage assisté — ultra simple", layout="wide")

st.title("🧮 Comptage assisté (dessiner 3 exemples) — générique")

# Réglages simples
with st.sidebar:
    st.header("Réglages")
    thr_factor = st.slider("Seuil (sans négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais pos/neg (avec négatifs)", 0.5, 1.5, 1.0, 0.05)
    open_ksize = st.slider("Ouverture", 1, 15, 3, 1)
    close_ksize= st.slider("Fermeture", 1, 15, 3, 1)
    min_area   = st.slider("Aire min (px²)", 10, 5000, 120, 10)
    max_area   = st.slider("Aire max (px²)", 1000, 200000, 30000, 500)
    rm_radius  = st.slider("Rayon suppression (px)", 5, 60, 20, 1)
    item_name  = st.text_input("Nom validé (ex: crevette, riz...)", "objet")

# 1) Upload
up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
img = Image.open(up).convert("RGB") if up else None

if img is None:
    st.info("1) Charge la photo. 2) **Peins** quelques traits verts sur 3 items (ou plus). 3) (Option) peins des traits rouges sur le fond. 4) Clique **Compter**.")
    st.stop()

W, H = img.width, img.height
st.image(img, caption="Image d'entrée", use_column_width=True)

# 2) Peinture POSITIVE (objets) puis NEGATIVE (fond)
st.subheader("Exemples utilisateur")
cpos, cneg = st.columns(2)
with cpos:
    st.caption("Traits OBJET (VERT) — peins sur 3 items (ou +)")
    can_pos = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#00FF00",
        background_image=img,
        update_streamlit=True,
        height=H, width=W, drawing_mode="freedraw", key="can_pos"
    )
with cneg:
    st.caption("Traits FOND (ROUGE) — optionnel")
    can_neg = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#FF0000",
        background_image=img,
        update_streamlit=True,
        height=H, width=W, drawing_mode="freedraw", key="can_neg"
    )

pos_mask = mask_from_strokes(can_pos.json_data, H, W)
neg_mask = mask_from_strokes(can_neg.json_data, H, W)

# 3) Comptage
if st.button("🚀 Compter"):
    t0 = time.time()
    img_cv = pil_to_cv(img)
    mask = segment_from_seeds(
        img_cv, pos_mask, neg_mask,
        thr_factor=thr_factor, pos_bias=pos_bias,
        open_ksize=open_ksize, close_ksize=close_ksize
    )
    points_auto = count_from_mask(mask, min_area=min_area, max_area=max_area)
    overlay_auto = overlay_points_pil(img, points_auto, (0,255,0), radius=6)
    dt = (time.time()-t0)*1000

    st.session_state["mask"] = mask
    st.session_state["points_auto"] = points_auto
    st.session_state["overlay_auto"] = overlay_auto
    st.session_state["img"] = img
    st.session_state["W"], st.session_state["H"] = W, H
    st.success(f"Compte auto: {len(points_auto)}")
    st.caption(f"Temps de traitement ~ {dt:.0f} ms")
    st.image(overlay_auto, caption="Overlay comptage automatique", use_column_width=True)

# 4) Corrections manuelles (ajout / suppression)
if "points_auto" in st.session_state:
    img = st.session_state["img"]
    W,H = st.session_state["W"], st.session_state["H"]
    base_overlay = st.session_state["overlay_auto"]

    st.subheader("Corrections (option)")
    ca, cb = st.columns(2)
    with ca:
        st.caption("Ajouter des points MANQUANTS (VERT) — dessine de petits cercles")
        can_add = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#00FF00",
            background_image=base_overlay,
            update_streamlit=True,
            height=H, width=W, drawing_mode="circle", key="can_add"
        )
    with cb:
        st.caption("Supprimer des faux positifs (ROUGE) — dessine de petits cercles")
        can_rem = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#FF0000",
            background_image=base_overlay,
            update_streamlit=True,
            height=H, width=W, drawing_mode="circle", key="can_rem"
        )

    # extraire centres des cercles ajoutés/supprimés
    def extract_circle_centers(json_data):
        pts=[]
        if not json_data or "objects" not in json_data: return pts
        for o in json_data["objects"]:
            if o.get("type")=="circle":
                cx = int(o["left"] + o["rx"])
                cy = int(o["top"]  + o["ry"])
                pts.append((cx,cy))
        return pts

    add_pts = extract_circle_centers(can_add.json_data)
    rem_pts = extract_circle_centers(can_rem.json_data)

    # appliquer corrections
    detected = list(st.session_state["points_auto"])
    # suppression: pour chaque point rouge -> retirer la détection la plus proche dans rm_radius
    for (rx,ry) in rem_pts:
        idx, d = nearest_index(detected, rx, ry)
        if idx is not None and d <= rm_radius and len(detected)>0:
            detected.pop(idx)
    # ajout: concaténer (éviter doublons trop proches)
    for (ax,ay) in add_pts:
        idx, d = nearest_index(detected, ax, ay)
        if d > rm_radius:
            detected.append((ax,ay))

    final_overlay = overlay_points_pil(img, detected, (0,255,0), radius=6)
    st.metric("Compte final", len(detected))
    st.image(final_overlay, caption="Overlay final (après corrections)", use_column_width=True)

    # 5) Rapport simple (image + nom validé)
    st.subheader("Rapport simple")
    validated_name = st.text_input("Nom validé à afficher", value=st.session_state.get("validated_name", ""), key="validated_name")
    # dessiner le nom validé en légende (option : on le garde à part)
    buf = io.BytesIO()
    final_overlay.save(buf, format="PNG")
    st.download_button("⬇️ Télécharger l'image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"rapport_{validated_name or 'compte'}.png", mime="image/png")

    # petit CSV d'une ligne
    df = pd.DataFrame([{
        "nom_valide": validated_name or "objet",
        "compte_final": len(detected),
        "compte_auto": len(st.session_state["points_auto"])
    }])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger le résumé (CSV)", data=csv_bytes,
                       file_name=f"rapport_{validated_name or 'compte'}.csv", mime="text/csv")
