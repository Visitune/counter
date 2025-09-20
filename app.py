import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import io, time, cv2, re
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

def resize_max_side(img_pil, max_side=1024):
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil
    s = max_side / max(w, h)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return img_pil.resize((int(w*s), int(h*s)), resample)

def parse_rgb(s):
    if isinstance(s, str) and s.startswith("#") and len(s)==7:
        return tuple(int(s[i:i+2],16) for i in (1,3,5))
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", str(s))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def is_green(s):
    r,g,b = parse_rgb(s) or (0,0,0)
    return g>=200 and r<=80 and b<=80

def is_red(s):
    r,g,b = parse_rgb(s) or (0,0,0)
    return r>=200 and g<=80 and b<=80

def masks_from_canvas(json_data, H, W):
    pos = np.zeros((H,W), np.uint8)
    neg = np.zeros((H,W), np.uint8)
    if not json_data or "objects" not in json_data:
        return pos, neg
    for o in json_data["objects"]:
        if o.get("type")!="path": 
            continue
        stroke = o.get("stroke") or o.get("strokeColor") or ""
        path = o.get("path", [])
        if not path: 
            continue
        pts=[]
        for seg in path:
            if len(seg)>=3:
                _,x,y = seg[:3]
                pts.append((int(x),int(y)))
        if len(pts)>=2:
            thickness = max(1, int(o.get("strokeWidth",10)))
            target = pos if is_green(stroke) else (neg if is_red(stroke) else None)
            if target is None: 
                continue
            cv2.polylines(target, [np.array(pts,np.int32)], False, 255, thickness=thickness)
    return pos, neg

def segment_from_seeds(img_bgr, pos_mask, neg_mask, thr_factor=3.0, pos_bias=1.0,
                       open_ksize=3, close_ksize=3):
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    pos_idx = np.where(pos_mask.reshape(-1)>0)[0]
    neg_idx = np.where(neg_mask.reshape(-1)>0)[0]
    if len(pos_idx)<10:
        return np.zeros((H,W), np.uint8)

    mu_pos = lab[pos_idx].mean(axis=0)
    if len(neg_idx)>=10:
        mu_neg = lab[neg_idx].mean(axis=0)
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        dneg = np.linalg.norm(lab - mu_neg, axis=1)
        pred = (dpos < (pos_bias*dneg)).astype(np.uint8)
    else:
        dpos = np.linalg.norm(lab - mu_pos, axis=1)
        ref = np.median(dpos[pos_idx]) if len(pos_idx) else dpos.mean()
        pred = (dpos < (thr_factor*ref)).astype(np.uint8)

    mask = (pred.reshape(H,W)*255).astype(np.uint8)
    if open_ksize>1:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_ksize,open_ksize))
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize>1:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_ksize,close_ksize))
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def count_from_mask(mask, min_area=80, max_area=50000):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts=[]
    for i in range(1,num):
        area=stats[i, cv2.CC_STAT_AREA]
        if min_area<=area<=max_area:
            cx,cy=centroids[i]
            pts.append((float(cx), float(cy)))
    return pts

def nearest_index(points, qx, qy):
    if not points: return None, 1e9
    dmin, kmin = 1e9, None
    for k,(x,y) in enumerate(points):
        d=(x-qx)**2 + (y-qy)**2
        if d<dmin: dmin,kmin=d,k
    return kmin, dmin**0.5

# ---------------- App ----------------
st.set_page_config(page_title="Comptage assisté — simple", layout="wide")
st.title("🧮 Comptage assisté (peindre 3 exemples) — générique")

with st.sidebar:
    st.header("Réglages")
    paint_mode = st.radio("Couleur du trait", ["Objet (VERT)", "Fond (ROUGE)"], horizontal=True)
    stroke_color = "#00FF00" if "VERT" in paint_mode else "#FF0000"
    thr_factor = st.slider("Seuil (sans négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais pos/neg (avec négatifs)", 0.5, 1.5, 1.0, 0.05)
    open_ksize = st.slider("Ouverture", 1, 15, 3, 1)
    close_ksize= st.slider("Fermeture", 1, 15, 3, 1)
    min_area   = st.slider("Aire min (px²)", 10, 5000, 120, 10)
    max_area   = st.slider("Aire max (px²)", 1000, 200000, 30000, 500)
    rm_radius  = st.slider("Rayon suppression (px)", 5, 60, 20, 1)

# Upload
up = st.file_uploader("Dépose une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("1) Charge la photo. 2) Peins des traits VERTS sur 3 items (ou +), et éventuellement ROUGES sur le fond. 3) Clique **Compter**.")
    st.stop()

# Image -> RGB (pas RGBA) + réduction max 1024
orig = Image.open(up).convert("RGB")
disp = resize_max_side(orig, max_side=1024)
st.image(disp, caption="Image d'entrée", use_column_width=True)

# Gestion du reset canvas
if "canvas_key" not in st.session_state: st.session_state["canvas_key"]=0
if st.button("🧹 Effacer les traits"): st.session_state["canvas_key"] += 1

# Canvas (IMPORTANT : pas de height/width pour laisser le composant épouser l'image)
st.subheader("Exemples utilisateur (peindre ici)")
canvas = st_canvas(
    background_image=disp.copy(),      # PIL.Image RGB
    background_color="#00000000",      # transparent
    stroke_color=stroke_color,
    stroke_width=10,
    fill_color="rgba(0,0,0,0)",
    drawing_mode="freedraw",
    update_streamlit=True,
    display_toolbar=True,
    key=f"canvas_{st.session_state['canvas_key']}"
)

# Masques depuis les traits
H, W = disp.size[1], disp.size[0]
pos_mask, neg_mask = masks_from_canvas(canvas.json_data, H, W)

# Comptage
if st.button("🚀 Compter"):
    t0 = time.time()
    img_cv = pil_to_cv(disp)  # même échelle que le canvas
    mask = segment_from_seeds(img_cv, pos_mask, neg_mask,
                              thr_factor=thr_factor, pos_bias=pos_bias,
                              open_ksize=open_ksize, close_ksize=close_ksize)
    pts = count_from_mask(mask, min_area=min_area, max_area=max_area)
    overlay = overlay_points_pil(disp, pts, (0,255,0), radius=6)

    st.session_state["disp"] = disp
    st.session_state["pts"] = pts
    st.session_state["overlay"] = overlay

    st.success(f"Compte auto: {len(pts)}")
    st.caption(f"Temps de traitement ~ {(time.time()-t0)*1000:.0f} ms")
    st.image(overlay, caption="Overlay comptage automatique", use_column_width=True)

# Corrections rapides
if "pts" in st.session_state:
    disp_rgb = st.session_state["disp"]
    overlay = st.session_state["overlay"]

    st.subheader("Corrections (option)")
    colA, colB = st.columns(2)
    with colA:
        st.caption("Ajouter (VERT) — dessine de petits cercles")
        can_add = st_canvas(background_image=overlay.copy(),
                            drawing_mode="circle", stroke_width=12,
                            stroke_color="#00FF00", fill_color="rgba(0,0,0,0)",
                            update_streamlit=True, key="can_add")
    with colB:
        st.caption("Supprimer (ROUGE) — dessine de petits cercles")
        can_rem = st_canvas(background_image=overlay.copy(),
                            drawing_mode="circle", stroke_width=12,
                            stroke_color="#FF0000", fill_color="rgba(0,0,0,0)",
                            update_streamlit=True, key="can_rem")

    def get_circle_centers(js):
        pts=[]
        if js and "objects" in js:
            for o in js["objects"]:
                if o.get("type")=="circle":
                    pts.append((int(o["left"]+o["rx"]), int(o["top"]+o["ry"])))
        return pts

    add_pts = get_circle_centers(can_add.json_data)
    rem_pts = get_circle_centers(can_rem.json_data)

    detected = list(st.session_state["pts"])
    for (rx,ry) in rem_pts:
        idx, d = nearest_index(detected, rx, ry)
        if idx is not None and d <= rm_radius and len(detected)>0:
            detected.pop(idx)
    for (ax,ay) in add_pts:
        idx, d = nearest_index(detected, ax, ay)
        if d > rm_radius:
            detected.append((ax,ay))

    final_overlay = overlay_points_pil(disp_rgb, detected, (0,255,0), radius=6)
    st.metric("Compte final", len(detected))
    st.image(final_overlay, caption="Overlay final (après corrections)", use_column_width=True)

    st.subheader("Rapport")
    name = st.text_input("Nom validé à afficher", value="")
    buf = io.BytesIO()
    final_overlay.save(buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=buf.getvalue(),
                       file_name=f"rapport_{name or 'compte'}.png", mime="image/png")
    df = pd.DataFrame([{"nom_valide": name or "objet",
                        "compte_final": len(detected),
                        "compte_auto": len(st.session_state['pts'])}])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"rapport_{name or 'compte'}.csv", mime="text/csv")
