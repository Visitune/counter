import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io, cv2, time

# ---------- Utils ----------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def resize_max(img_pil, max_side=1200):
    w, h = img_pil.size
    if max(w, h) <= max_side:
        return img_pil, 1.0
    s = max_side / max(w, h)
    return img_pil.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS), s

def draw_points_pil(image_pil, points, color=(0,255,0), radius=6):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def make_template_for_paint(img):
    """Ajoute une barre d'instructions en haut de l'image à annoter (VERT=objet, ROUGE=fond)."""
    w, h = img.size
    banner_h = max(40, h//16)
    tmpl = Image.new("RGB", (w, h + banner_h), color=(255,255,255))
    tmpl.paste(img, (0, banner_h))
    d = ImageDraw.Draw(tmpl)
    text = "Dessinez VERT sur 3–5 objets • (Option) dessinez ROUGE sur le fond • Enregistrez puis ré-uploadez ici"
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, banner_h//3))
    except:
        font = ImageFont.load_default()
    d.text((10, 10), text, fill=(0,0,0), font=font)
    # Légende
    d.rectangle([w-240, 5, w-10, banner_h-5], outline=(0,0,0), width=1)
    d.ellipse([w-230, 10, w-210, 30], fill=(0,255,0)); d.text((w-205, 12), "OBJET (VERT)", fill=(0,0,0), font=font)
    d.ellipse([w-230, 30, w-210, 50], fill=(255,0,0)); d.text((w-205, 32), "FOND (ROUGE)", fill=(0,0,0), font=font)
    return tmpl

def extract_pos_neg_masks(annotated, top_banner=True):
    """Extrait deux masques à partir d'un PNG annoté avec vert/rouge."""
    img = annotated.convert("RGB")
    w,h = img.size
    y0 = (h//16) if top_banner else 0
    crop = img.crop((0, y0, w, h))  # on ignore la bannière
    arr = np.array(crop)
    R,G,B = arr[:,:,0].astype(np.int16), arr[:,:,1].astype(np.int16), arr[:,:,2].astype(np.int16)
    # tolérances (verts “suffisamment verts”, rouges “suffisamment rouges”)
    pos = (G - np.maximum(R,B) >= 60) & (G >= 160)
    neg = (R - np.maximum(G,B) >= 60) & (R >= 160)
    pos_mask = np.uint8(pos) * 255
    neg_mask = np.uint8(neg) * 255
    return pos_mask, neg_mask

def segment_from_seeds(img_bgr, pos_mask, neg_mask,
                       thr_factor=3.0, pos_bias=1.0,
                       open_k=3, close_k=3):
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
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def count_components(mask, min_area=80, max_area=100000):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area <= a <= max_area:
            cx, cy = cents[i]
            pts.append((float(cx), float(cy)))
    return pts

# ---------- App ----------
st.set_page_config(page_title="Comptage (méthode gabarit robuste)", layout="wide")
st.title("🧮 Comptage simple et robuste (annoter dans un éditeur externe)")

with st.sidebar:
    st.header("Paramètres")
    thr_factor = st.slider("Seuil (si pas de négatifs)", 1.0, 6.0, 3.0, 0.1)
    pos_bias   = st.slider("Biais pos/neg (si négatifs présents)", 0.5, 1.5, 1.0, 0.05)
    open_k     = st.slider("Ouverture (morpho)", 1, 15, 3, 1)
    close_k    = st.slider("Fermeture (morpho)", 1, 15, 3, 1)
    min_area   = st.slider("Aire min (px²)", 10, 20000, 120, 10)
    max_area   = st.slider("Aire max (px²)", 100, 300000, 30000, 500)
    correction = st.number_input("Correction finale (+/-)", -9999, 9999, 0)

up = st.file_uploader("1) Déposer une image (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("Dépose une image. Puis : Télécharger gabarit → dessiner VERT/ROUGE dans un éditeur → Ré-uploader le PNG annoté → Compter.")
    st.stop()

orig = Image.open(up).convert("RGB")
disp, scale = resize_max(orig, 1200)
st.image(disp, caption="Image (échelle d'annotation)", use_column_width=True)

# 2) Gabarit à annoter
st.subheader("2) Gabarit (à annoter dans Paint/Gimp, etc.)")
tmpl = make_template_for_paint(disp)
buf = io.BytesIO(); tmpl.save(buf, format="PNG")
st.download_button("⬇️ Télécharger gabarit (PNG)", data=buf.getvalue(),
                   file_name="gabarit_annotation.png", mime="image/png")
st.caption("Ouvre ce PNG, peins **VERT** sur 3–5 objets et (option) **ROUGE** sur le fond, enregistre puis charge-le ci-dessous.")

ann = st.file_uploader("3) Charger le PNG annoté (le même gabarit modifié)", type=["png"])
if not ann:
    st.stop()

annotated = Image.open(ann).convert("RGB")
st.image(annotated, caption="PNG annoté (tel qu'uploadé)", use_column_width=True)

# 4) Segmentation + Comptage
if st.button("🚀 Compter"):
    t0 = time.time()
    pos_mask, neg_mask = extract_pos_neg_masks(annotated, top_banner=True)
    img_cv = pil_to_cv(disp)
    seg = segment_from_seeds(img_cv, pos_mask, neg_mask,
                             thr_factor=thr_factor, pos_bias=pos_bias,
                             open_k=open_k, close_k=close_k)
    pts = count_components(seg, min_area=min_area, max_area=max_area)
    overlay = draw_points_pil(disp, pts, (0,255,0), radius=6)
    auto = len(pts)
    final = max(0, auto + int(correction))

    st.success(f"Compte auto: {auto}  •  Temps ~ {(time.time()-t0)*1000:.0f} ms")
    st.metric("Compte final (avec correction)", final)
    st.image(overlay, caption="Overlay (points sur items comptés)", use_column_width=True)

    # 5) Exports
    st.subheader("5) Export rapport")
    nom = st.text_input("Nom validé (ex: crevettes, riz…)", value="")
    img_buf = io.BytesIO(); overlay.save(img_buf, format="PNG")
    st.download_button("⬇️ Image annotée (PNG)", data=img_buf.getvalue(),
                       file_name=f"rapport_{nom or 'compte'}.png", mime="image/png")
    df = pd.DataFrame([{
        "nom_valide": nom or "objet",
        "compte_auto": auto,
        "correction": int(correction),
        "compte_final": final
    }])
    st.download_button("⬇️ Résumé (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"rapport_{nom or 'compte'}.csv", mime="text/csv")
