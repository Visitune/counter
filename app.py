import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import io, json, time, cv2
from math import sqrt

# ---------------- Utils ----------------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points_pil(image_pil, points, color=(0,255,0), radius=6):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def resize_for_canvas(img_pil, max_side=1280):
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil, 1.0
    s = max_side / max(w, h)
    img2 = img_pil.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img2, s

def mask_from_strokes(canvas_json, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not canvas_json or "objects" not in canvas_json:
        return mask
    for obj in canvas_json["objects"]:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            if not path: 
                continue
            pts = []
            for seg in path:
                if len(seg) >= 3:
                    _, x, y = seg[:3]
                    pts.append((int(x), int(y)))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255,
                              thickness=max(1, int(obj.get("strokeWidth", 2))))
    return mask

def nearest_index(points, qx, qy):
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = (x-qx)**2 + (y-qy)**2
        if d < dmin: dmin, kmin = d, k
    return kmin, dmin**0.5

# ---- Segmentation guidée par exemples (Lab) ----
def segment_from_seeds(img_bgr, pos_mask, neg_mask, thr_factor=3.0, pos_bias=1.0,
                       open_ksize=3, close_ksize=3):
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)

    pos_idx = np.where(pos_mask.reshape(-1) > 0)[0]
    neg_idx = np.where(neg_mask.reshape(-1) > 0)[0]
    if len(pos_idx) < 10:
        return np.zeros((H,W), np.uint8)

    mu_pos = lab[pos_idx].mean(axis=0)
    if len(neg_idx) >= 10:
        mu_neg = lab[neg_idx].mean(axis=0)
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        dneg = np.linalg.norm(lab - mu_neg, axis=1)
        pred = (dpos < (pos_bias * dneg)).astype(np.uint8)
    else:
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        ref = np.median(dpos[pos_idx]) if len(pos_idx) else dpos.mean()
        pred = (dpos < (thr_factor * ref)).astype(np.uint8)

    mask = (pred.reshape(H,W) * 255).astype(np.uint8)
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

# ---------------- App ----------------
st.set_page_config(page_title="Comptage assisté — très simple", layout="wide")
st.title("🧮 Comptage assisté (peindre 3 exemples) — générique")

with st.sidebar:
    st.header("Réglages")
    thr_factor = st.slider("Seuil (sans négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais pos/neg (avec négatifs)", 0.5, 1.5, 1.0, 0.05)
    open_ksize = st.slider("Ouverture", 1, 15, 3, 1)
    close_ksize= st.slider("Fermeture", 1, 15, 3, 1)
    min_area   = st.slider("Aire min (px²)", 10, 5000, 120, 10)
    max_area   = st.slider("Aire max (px²)", 1000, 200000, 30000, 500)
    rm_radius  = st.slider("Rayon suppression (px)", 5, 60, 20, 1)
    item_name  = st.text_input("Nom validé", "objet")

# 1) Upload
up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("1) Charge la photo. 2) Peins quelques traits verts sur 3 items (ou +). 3) (Option) peins des traits rouges sur le fond. 4) Clique **Compter**.")
    st.stop()

orig = Image.open(up).convert("RGB")
disp, scale = resize_for_canvas(orig, max_side=1280)   # <— clé : image redimensionnée pour le canvas
W, H = disp.size

st.image(disp, caption="Image d'entrée", use_column_width=True)

# 2) Canvases avec l'image *affichée* (numpy array)
st.subheader("Exemples utilisateur")
cpos, cneg = st.columns(2)
with cpos:
    st.caption("Traits OBJET (VERT) — peins sur 3 items (ou plus)")
    can_pos = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#00FF00",
        background_image=np.array(disp),   # <— numpy array, pas PIL
        update_streamlit=True,
        height=H, width=W,
        drawing_mode="freedraw",
        key="can_pos"
    )
with cneg:
    st.caption("Traits FOND (ROUGE) — optionnel")
    can_neg = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#FF0000",
        background_image=np.array(disp),   # <— numpy array, pas PIL
        update_streamlit=True,
        height=H, width=W,
        drawing_mode="freedraw",
        key="can_neg"
    )

pos_mask = mask_from_strokes(can_pos.json_data, H, W)
neg_mask = mask_from_strokes(can_neg.json_data, H, W)

# 3) Comptage
if st.button("🚀 Compter"):
    t0 = time.time()
    img_cv = pil_to_cv(disp)  # on travaille à l'échelle affichée (simple & robuste pour le canvas)
    mask = segment_from_seeds(
        img_cv, pos_mask, neg_mask,
        thr_factor=thr_factor, pos_bias=pos_bias,
        open_ksize=open_ksize, close_ksize=close_ksize
    )
    points_auto = count_from_mask(mask, min_area=min_area, max_area=max_area)
    overlay_auto = overlay_points_pil(disp, points_auto, (0,255,0), radius=6)
    st.session_state["disp"] = disp
    st.session_state["points_auto"] = points_auto
    st.session_state["overlay_auto"] = overlay_auto
    st.success(f"Compte auto: {len(points_auto)}")
    st.caption(f"Temps de traitement ~ {(time.time()-t0)*1000:.0f} ms")
    st.image(overlay_auto, caption="Overlay comptage automatique", use_column_width=True)

# 4) Corrections
if "points_auto" in st.session_state:
    disp = st.session_state["disp"]
    base_overlay = st.session_state["overlay_auto"]

    st.subheader("Corrections (option)")
    ca, cb = st.columns(2)
    with ca:
        st.caption("Ajouter des points MANQUANTS (VERT) — dessine de petits cercles")
        can_add = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#00FF00",
            background_image=np.array(base_overlay),
            update_streamlit=True,
            height=H, width=W,
            drawing_mode="circle",
            key="can_add"
        )
    with cb:
        st.caption("Supprimer des faux positifs (ROUGE) — dessine de petits cercles")
        can_rem = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#FF0000",
            background_image=np.array(base_overlay),
            update_streamlit=True,
            height=H, width=W,
            drawing_mode="circle",
            key="can_rem"
        )

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

    def apply_corrections(points, add_pts, rem_pts, rm_radius):
        pts = list(points)
        for (rx,ry) in rem_pts:
            idx, d = nearest_index(pts, rx, ry)
            if idx is not None and d <= rm_radius and len(pts)>0:
                pts.pop(idx)
        for (ax,ay) in add_pts:
            idx, d = nearest_index(pts, ax, ay)
            if d > rm_radius:
                pts.append((ax,ay))
        return pts

    detected = apply_corrections(st.session_state["points_auto"], add_pts, rem_pts, rm_radius)
    final_overlay = overlay_points_pil(disp, detected, (0,255,0), radius=6)
    st.metric("Compte final", len(detected))
    st.image(final_overlay, caption="Overlay final (après corrections)", use_column_width=True)

    # 5) Rapport simple
    st.subheader("Rapport simple")
    validated_name = st.text_input("Nom validé à afficher", value=st.session_state.get("validated_name",""))
    buf = io.BytesIO()
    final_overlay.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"rapport_{validated_name or 'compte'}.png", mime="image/png")
    df = pd.DataFrame([{
        "nom_valide": validated_name or "objet",
        "compte_final": len(detected),
        "compte_auto": len(st.session_state['points_auto'])
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"rapport_{validated_name or 'compte'}.csv", mime="text/csv")
