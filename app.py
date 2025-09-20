import streamlit as st
from PIL import Image
import numpy as np
import cv2, io, time
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ---------- Helpers ----------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def resize_max(img_pil, max_side=1024):
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil, 1.0
    s = max_side / max(w, h)
    img2 = img_pil.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)
    return img2, s

def draw_points_pil(image_pil, points, color=(0,255,0), radius=6):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def mask_from_clicks(h, w, pts, r=6):
    """Disques autour des clics pour constituer les graines."""
    m = np.zeros((h,w), np.uint8)
    for (x,y) in pts:
        cv2.circle(m, (int(x),int(y)), r, 255, -1)
    return m

def segment_from_seeds(img_bgr, pos_mask, neg_mask, thr_factor=3.0, pos_bias=1.0,
                       open_ksize=3, close_ksize=3):
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    pos_idx = np.where(pos_mask.reshape(-1) > 0)[0]
    neg_idx = np.where(neg_mask.reshape(-1) > 0)[0]
    if len(pos_idx) < 5:  # 5 clics mini (ou r plus grand)
        return np.zeros((H,W), np.uint8)

    mu_pos = lab[pos_idx].mean(axis=0)
    if len(neg_idx) >= 5:
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

def count_cc(mask, min_area=80, max_area=50000):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts=[]
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))
    return pts

def nearest_index(points, qx, qy):
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = (x-qx)**2 + (y-qy)**2
        if d < dmin: dmin, kmin = d, k
    return kmin, dmin**0.5

def fig_image_with_points(pil_img, pos_pts=None, neg_pts=None, det_pts=None, height_limit=700):
    w,h = pil_img.size
    fig = go.Figure()
    fig.add_layout_image(
        dict(source=pil_img, x=0, y=0, sizex=w, sizey=h, xref="x", yref="y", layer="below")
    )
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1)

    if pos_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pos_pts], y=[p[1] for p in pos_pts],
            mode="markers", marker=dict(color="lime", size=10), name="OBJET"
        ))
    if neg_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in neg_pts], y=[p[1] for p in neg_pts],
            mode="markers", marker=dict(color="red", size=10), name="FOND"
        ))
    if det_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in det_pts], y=[p[1] for p in det_pts],
            mode="markers", marker=dict(color="cyan", size=8, symbol="x"), name="Détections"
        ))

    # hauteur raisonnable
    height = min(height_limit, int(h * (height_limit / h))) if h > height_limit else h
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height)
    return fig

# ---------- App ----------
st.set_page_config(page_title="Comptage — clics simples", layout="wide")
st.title("🧮 Comptage par clics (générique)")

with st.sidebar:
    st.header("Réglages")
    click_mode = st.radio("Mode clic", ["Ajouter OBJET (vert)", "Ajouter FOND (rouge)"], horizontal=False)
    seed_radius = st.slider("Rayon graine (px)", 3, 25, 8, 1)
    thr_factor  = st.slider("Seuil (sans négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias    = st.slider("Biais pos/neg (avec négatifs)", 0.5, 1.5, 1.0, 0.05)
    open_k      = st.slider("Ouverture", 1, 15, 3, 1)
    close_k     = st.slider("Fermeture", 1, 15, 3, 1)
    min_area    = st.slider("Aire min (px²)", 10, 5000, 120, 10)
    max_area    = st.slider("Aire max (px²)", 500, 200000, 30000, 500)
    rm_radius   = st.slider("Rayon correction (px)", 5, 60, 20, 1)

up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("Charge une image, puis **clique** sur 3+ objets (verts) et (option) quelques fonds (rouges).")
    st.stop()

orig = Image.open(up).convert("RGB")
disp, scale = resize_max(orig, max_side=1024)
W, H = disp.size
st.caption("Image d'entrée (redimensionnée pour l'annotation)")
st.image(disp, use_column_width=True)

# état
if "pos_pts" not in st.session_state: st.session_state.pos_pts = []
if "neg_pts" not in st.session_state: st.session_state.neg_pts = []
if "det_pts" not in st.session_state: st.session_state.det_pts = None
if "overlay" not in st.session_state: st.session_state.overlay = None

# --- CLICS D'ÉCHANTILLONNAGE ---
st.subheader("Échantillons (cliquez sur l'image)")
fig = fig_image_with_points(disp, st.session_state.pos_pts, st.session_state.neg_pts)
click = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=fig.layout.height)

if click:
    x = float(click[0]["x"]); y = float(click[0]["y"])
    if "OBJET" in click_mode:
        st.session_state.pos_pts.append((x,y))
    else:
        st.session_state.neg_pts.append((x,y))

c1, c2, c3 = st.columns(3)
if c1.button("↺ Réinitialiser points"):
    st.session_state.pos_pts = []
    st.session_state.neg_pts = []
    st.session_state.det_pts = None
    st.session_state.overlay = None

with c2:
    if st.button("🚀 Compter"):
        t0 = time.time()
        pos_mask = mask_from_clicks(H, W, st.session_state.pos_pts, r=seed_radius)
        neg_mask = mask_from_clicks(H, W, st.session_state.neg_pts, r=seed_radius)
        mask = segment_from_seeds(
            pil_to_cv(disp), pos_mask, neg_mask,
            thr_factor=thr_factor, pos_bias=pos_bias,
            open_ksize=open_k, close_ksize=close_k
        )
        det = count_cc(mask, min_area=min_area, max_area=max_area)
        st.session_state.det_pts = det
        st.session_state.overlay = draw_points_pil(disp, det, (0,255,0), radius=6)
        st.success(f"Compte auto: {len(det)}  •  {(time.time()-t0)*1000:.0f} ms")

# --- CORRECTION ---
if st.session_state.det_pts is not None:
    st.subheader("Corrections")
    mode_corr = st.radio("Mode correction", ["Ajouter point", "Supprimer point"], horizontal=True)
    fig2 = fig_image_with_points(disp, det_pts=st.session_state.det_pts)
    click2 = plotly_events(fig2, click_event=True, hover_event=False, select_event=False, override_height=fig2.layout.height)

    det = list(st.session_state.det_pts)
    if click2:
        x = float(click2[0]["x"]); y = float(click2[0]["y"])
        if mode_corr == "Supprimer point":
            idx, d = nearest_index(det, x, y)
            if idx is not None and d <= rm_radius and len(det) > 0:
                det.pop(idx)
        else:
            # ajout si pas trop proche d'un point existant
            idx, d = nearest_index(det, x, y)
            if d > rm_radius:
                det.append((x,y))

        st.session_state.det_pts = det
        st.session_state.overlay = draw_points_pil(disp, det, (0,255,0), radius=6)

    st.metric("Compte final", len(st.session_state.det_pts))
    st.image(st.session_state.overlay, caption="Overlay final", use_column_width=True)

    st.subheader("Rapport")
    nom = st.text_input("Nom validé (ex: crevette, riz...)", value="")
    buf = io.BytesIO()
    st.session_state.overlay.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"rapport_{nom or 'compte'}.png", mime="image/png")
    df = pd.DataFrame([{
        "nom_valide": nom or "objet",
        "compte_final": len(st.session_state.det_pts),
        "compte_auto": len(st.session_state.det_pts)  # identique après corrections
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"rapport_{nom or 'compte'}.csv", mime="text/csv")
