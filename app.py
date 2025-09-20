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

def resize_max_side(img_pil, max_side=800):  # Taille réduite pour éviter les problèmes
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil, 1.0
    s = max_side / max(w, h)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    new_img = img_pil.resize((int(w*s), int(h*s)), resample)
    return new_img, s

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

# Image processing
orig = Image.open(up).convert("RGB")
disp, scale_factor = resize_max_side(orig, max_side=800)
W, H = disp.size

st.image(disp, caption=f"Image d'entrée ({W}x{H})", use_column_width=True)

# Gestion du reset canvas
if "canvas_key" not in st.session_state: 
    st.session_state["canvas_key"] = 0

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Exemples utilisateur (peindre ici)")
with col2:
    if st.button("🧹 Effacer les traits"): 
        st.session_state["canvas_key"] += 1
        st.rerun()

# Canvas principal - sans background_image pour éviter les erreurs
try:
    canvas = st_canvas(
        stroke_color=stroke_color,
        stroke_width=8,
        fill_color="rgba(0,0,0,0)",
        drawing_mode="freedraw",
        update_streamlit=True,
        display_toolbar=True,
        height=H,
        width=W,
        key=f"canvas_{st.session_state['canvas_key']}"
    )
    
    # Afficher l'image séparément comme référence
    st.caption("👆 Dessinez vos traits sur la zone ci-dessus en regardant l'image de référence ci-dessous")
    
except Exception as e:
    st.error(f"Erreur canvas: {str(e)}")
    st.stop()

# Debug info
if canvas.json_data:
    num_objects = len(canvas.json_data.get("objects", []))
    if num_objects > 0:
        st.success(f"✅ {num_objects} trait(s) dessiné(s)")

# Masques depuis les traits
pos_mask, neg_mask = masks_from_canvas(canvas.json_data, H, W)

# Debug masks
if np.sum(pos_mask) > 0:
    st.success(f"✅ Traits verts détectés ({np.sum(pos_mask > 0)} pixels)")
if np.sum(neg_mask) > 0:
    st.info(f"ℹ️ Traits rouges détectés ({np.sum(neg_mask > 0)} pixels)")

# Comptage
if st.button("🚀 Compter", type="primary"):
    if np.sum(pos_mask) == 0:
        st.error("❌ Aucun trait vert détecté ! Dessinez d'abord sur quelques objets.")
    else:
        with st.spinner("Traitement en cours..."):
            t0 = time.time()
            img_cv = pil_to_cv(disp)
            mask = segment_from_seeds(img_cv, pos_mask, neg_mask,
                                      thr_factor=thr_factor, pos_bias=pos_bias,
                                      open_ksize=open_ksize, close_ksize=close_ksize)
            pts = count_from_mask(mask, min_area=min_area, max_area=max_area)
            
            # Convertir les points vers l'image originale
            if scale_factor != 1.0:
                pts_orig = [(x/scale_factor, y/scale_factor) for x, y in pts]
                overlay_orig = overlay_points_pil(orig, pts_orig, (0,255,0), radius=int(6/scale_factor))
            else:
                pts_orig = pts
                overlay_orig = overlay_points_pil(orig, pts, (0,255,0), radius=6)
            
            overlay_disp = overlay_points_pil(disp, pts, (0,255,0), radius=6)

            # Sauvegarder dans session state
            st.session_state["disp"] = disp
            st.session_state["orig"] = orig
            st.session_state["pts"] = pts
            st.session_state["pts_orig"] = pts_orig
            st.session_state["overlay_disp"] = overlay_disp
            st.session_state["overlay_orig"] = overlay_orig
            st.session_state["scale_factor"] = scale_factor

            dt = (time.time()-t0)*1000
            st.success(f"Compte détecté: {len(pts)}")
            st.caption(f"Temps de traitement: {dt:.0f} ms")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay_disp, caption="Résultat (version affichage)")
            with col2:
                # Afficher le masque de segmentation
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                mask_pil = cv_to_pil(mask_colored)
                st.image(mask_pil, caption="Masque de segmentation")

# Corrections rapides (version simplifiée sans canvas)
if "pts" in st.session_state:
    st.subheader("Corrections manuelles")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        add_count = st.number_input("Ajouter des objets", min_value=0, max_value=50, value=0, step=1)
    with col2:
        remove_count = st.number_input("Retirer des objets", min_value=0, max_value=len(st.session_state["pts"]), value=0, step=1)
    with col3:
        if st.button("Appliquer corrections"):
            current_count = len(st.session_state["pts"])
            final_count = max(0, current_count + add_count - remove_count)
            st.session_state["final_count"] = final_count
            
            # Créer une version corrigée (simulation)
            if final_count != current_count:
                st.success(f"Compte corrigé: {current_count} → {final_count}")
    
    # Affichage du résultat final
    final_count = st.session_state.get("final_count", len(st.session_state["pts"]))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Compte final", final_count)
        if final_count != len(st.session_state["pts"]):
            correction = final_count - len(st.session_state["pts"])
            st.metric("Correction appliquée", f"{correction:+d}")
    
    with col2:
        st.image(st.session_state["overlay_orig"], caption="Image finale (résolution originale)")

    # Rapport et export
    st.subheader("Export")
    name = st.text_input("Nom du produit", value="", placeholder="Ex: crevettes, riz...")
    
    if st.button("📄 Générer rapport"):
        # Export image
        buf_img = io.BytesIO()
        st.session_state["overlay_orig"].save(buf_img, format="PNG")
        
        # Export CSV
        df = pd.DataFrame([{
            "nom_produit": name or "produit",
            "fichier_image": up.name,
            "compte_automatique": len(st.session_state["pts"]),
            "compte_final": final_count,
            "correction": final_count - len(st.session_state["pts"]),
            "parametres": f"thr={thr_factor}, bias={pos_bias}, aire=[{min_area},{max_area}]"
        }])
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Image annotée (PNG)", 
                data=buf_img.getvalue(),
                file_name=f"comptage_{name or 'produit'}.png", 
                mime="image/png"
            )
        with col2:
            st.download_button(
                "⬇️ Rapport (CSV)", 
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"rapport_{name or 'produit'}.csv", 
                mime="text/csv"
            )
        
        # Afficher le rapport
        st.dataframe(df, use_container_width=True)
