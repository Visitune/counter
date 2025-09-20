import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from PIL import Image
import io, cv2, time

# ---------- Utils ----------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def resize_max_side(img_pil, max_side=1024):
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil
    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return img_pil.resize(new_size, resample)

def overlay_points(image_pil, points, color=(0,255,0), radius=6):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def nearest_index(points, qx, qy):
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d = (x-qx)**2 + (y-qy)**2
        if d < dmin:
            dmin, kmin = d, k
    return kmin, dmin**0.5

# ---------- Segmentation guidée par CLICS ----------
def segment_from_clicks(img_bgr, pos_pts, neg_pts, sample_r=5, thr_k=3.0, bias=1.0,
                        open_ksize=3, close_ksize=3):
    """
    img_bgr : image OpenCV (BGR)
    pos_pts, neg_pts : listes [(x,y), ...] en pixels
    sample_r : rayon (px) pour échantillonner des pixels autour de chaque clic
    thr_k    : facteur de seuil si pas de négatifs
    bias     : dpos < bias * dneg pour classer en positif quand négatifs présents
    """
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def collect_samples(points):
        vals = []
        for (x,y) in points:
            xi, yi = int(round(x)), int(round(y))
            x1, x2 = max(0, xi - sample_r), min(W, xi + sample_r + 1)
            y1, y2 = max(0, yi - sample_r), min(H, yi + sample_r + 1)
            patch = lab[y1:y2, x1:x2, :].reshape(-1,3)
            if patch.size:
                vals.append(patch)
        return np.concatenate(vals, axis=0) if vals else np.empty((0,3), np.float32)

    pos_samples = collect_samples(pos_pts)
    neg_samples = collect_samples(neg_pts)

    if pos_samples.shape[0] < 10:
        return np.zeros((H,W), np.uint8)  # pas assez d'info

    mu_pos = pos_samples.mean(axis=0)
    if neg_samples.shape[0] >= 10:
        mu_neg = neg_samples.mean(axis=0)
        dpos = np.linalg.norm(lab - mu_pos, axis=2)
        dneg = np.linalg.norm(lab - mu_neg, axis=2)
        mask = (dpos < (bias * dneg)).astype(np.uint8) * 255
    else:
        dpos = np.linalg.norm(lab - mu_pos, axis=2)
        ref = np.median(np.linalg.norm(pos_samples - mu_pos, axis=1))
        mask = (dpos < thr_k * ref).astype(np.uint8) * 255

    # Nettoyage morpho
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def count_components(mask, min_area=80, max_area=50000):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))
    return pts

# ---------- Plotly helpers ----------
def figure_with_image(img_pil, pos_pts=None, neg_pts=None, det_pts=None, title=""):
    """Affiche l'image en fond + couches de points (verts/rouges/détections)."""
    w, h = img_pil.size
    fig = go.Figure()
    fig.add_layout_image(
        dict(source=img_pil, xref="x", yref="y", x=0, y=h, sizex=w, sizey=h, layer="below")
    )
    # Axes orientés image
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1)

    if pos_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pos_pts], y=[p[1] for p in pos_pts],
            mode="markers", marker=dict(size=9, color="lime"), name="Exemples objet"
        ))
    if neg_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in neg_pts], y=[p[1] for p in neg_pts],
            mode="markers", marker=dict(size=9, color="red"), name="Exemples fond"
        ))
    if det_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in det_pts], y=[p[1] for p in det_pts],
            mode="markers", marker=dict(size=8, color="cyan"), name="Détections"
        ))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), title=title, dragmode="pan")
    return fig

# ---------- App ----------
st.set_page_config(page_title="Comptage assisté — clics", layout="wide")
st.title("🧮 Comptage assisté par CLICS (générique)")

with st.sidebar:
    st.header("Réglages")
    mode_click = st.radio("Mode de clic", ["Ajouter OBJET (vert)", "Ajouter FOND (rouge)",
                                            "Supprimer détection"], horizontal=False)
    thr_k     = st.slider("Seuil (sans négatifs)", 1.0, 6.0, 3.0, 0.1)
    bias      = st.slider("Biais pos/neg (avec négatifs)", 0.5, 1.5, 1.0, 0.05)
    open_k    = st.slider("Ouverture", 1, 15, 3, 1)
    close_k   = st.slider("Fermeture", 1, 15, 3, 1)
    min_area  = st.slider("Aire min (px²)", 10, 5000, 120, 10)
    max_area  = st.slider("Aire max (px²)", 1000, 200000, 30000, 500)
    rm_radius = st.slider("Rayon suppression (px)", 5, 60, 20, 1)

# Upload
up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("1) Charge la photo. 2) Clique 3–6 fois sur les OBJETS (verts), puis (option) 3–6 fois sur le FOND (rouges). 3) Clique **Compter**.")
    st.stop()

orig = Image.open(up).convert("RGB")
disp = resize_max_side(orig, 1024)
W, H = disp.size

# State
if "pos_pts" not in st.session_state: st.session_state.pos_pts = []
if "neg_pts" not in st.session_state: st.session_state.neg_pts = []
if "det_pts" not in st.session_state: st.session_state.det_pts = []

# Figure interactive
fig = figure_with_image(disp, st.session_state.pos_pts, st.session_state.neg_pts,
                        st.session_state.det_pts, title="Clique dans l'image (selon le mode)")
events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="fig1")

# Traiter les clics
if events:
    x, y = events[0]["x"], events[0]["y"]
    if mode_click.startswith("Ajouter OBJET"):
        st.session_state.pos_pts.append((x, y))
    elif mode_click.startswith("Ajouter FOND"):
        st.session_state.neg_pts.append((x, y))
    else:
        # supprimer la détection la plus proche
        idx, d = nearest_index(st.session_state.det_pts, x, y)
        if idx is not None and d <= rm_radius and st.session_state.det_pts:
            st.session_state.det_pts.pop(idx)

# Boutons de reset
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🧹 Vider OBJET"):
        st.session_state.pos_pts = []
with c2:
    if st.button("🧹 Vider FOND"):
        st.session_state.neg_pts = []
with c3:
    if st.button("🧹 Vider DÉTECTIONS"):
        st.session_state.det_pts = []

st.write(f"Exemples OBJET: {len(st.session_state.pos_pts)} • Exemples FOND: {len(st.session_state.neg_pts)}")

# Lancer le comptage
if st.button("🚀 Compter"):
    t0 = time.time()
    mask = segment_from_clicks(
        pil_to_cv(disp),
        st.session_state.pos_pts,
        st.session_state.neg_pts,
        sample_r=5, thr_k=thr_k, bias=bias,
        open_ksize=open_k, close_ksize=close_k
    )
    det_pts = count_components(mask, min_area=min_area, max_area=max_area)
    st.session_state.det_pts = det_pts

    st.success(f"Compte auto: {len(det_pts)}")
    st.caption(f"Temps de traitement ~ {(time.time()-t0)*1000:.0f} ms")

    # Afficher overlay
    fig2 = figure_with_image(disp, st.session_state.pos_pts, st.session_state.neg_pts,
                             st.session_state.det_pts, title="Résultat (clics rouge = suppression)")
    plotly_events(fig2, click_event=True, hover_event=False, select_event=False, key="fig2")

# Exports
if st.session_state.det_pts:
    overlay = overlay_points(disp, st.session_state.det_pts, (0,255,0), radius=6)
    st.image(overlay, caption="Overlay final", use_column_width=True)

    name = st.text_input("Nom validé à afficher (pour le nom de fichier)", value="")
    buf = io.BytesIO(); overlay.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"rapport_{name or 'compte'}.png", mime="image/png")
    df = pd.DataFrame([{
        "nom_valide": name or "objet",
        "compte_final": len(st.session_state.det_pts),
        "nb_exemples_objet": len(st.session_state.pos_pts),
        "nb_exemples_fond": len(st.session_state.neg_pts),
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"rapport_{name or 'compte'}.csv", mime="text/csv")
