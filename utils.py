# utils.py

import numpy as np
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import streamlit as st  # <--- CETTE LIGNE A ÉTÉ AJOUTÉE

# ================= Modèle d'IA (CLIP) =================
@st.cache_resource
def get_clip_model():
    """Charge le modèle CLIP une seule fois et le met en cache."""
    return SentenceTransformer('clip-ViT-B-32')

# ================= Conversion et Dessin =================
# ... (pil_to_cv, cv_to_pil restent identiques)

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
def overlay_objects(image_pil, objects, radius=8):
    """Dessine les objets avec des couleurs indiquant leur statut (bon/mauvais)."""
    img = pil_to_cv(image_pil.copy())
    for obj in objects:
        color = (0, 255, 0) if obj['is_counted'] else (180, 180, 180)
        cv2.circle(img, (int(obj['cx']), int(obj['cy'])), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)
    
# ================= Détection & Caractéristiques =================
def suppress_duplicates(objects, radius_factor=2.0):
    """Filtre les objets pour ne garder qu'un point par groupe proche."""
    if not objects: return []
    objects.sort(key=lambda o: o['area'], reverse=True)
    kept_objects = []
    suppressed_indices = set()
    for i, obj1 in enumerate(objects):
        if i in suppressed_indices: continue
        kept_objects.append(obj1)
        suppression_radius = np.sqrt(obj1['area'] / np.pi) * radius_factor
        for j, obj2 in enumerate(objects):
            if i == j or j in suppressed_indices: continue
            dist = np.sqrt((obj1['cx'] - obj2['cx'])**2 + (obj1['cy'] - obj2['cy'])**2)
            if dist < suppression_radius:
                suppressed_indices.add(j)
    return kept_objects

def detect_and_embed_candidates(img_pil, model, min_area=50):
    """Détecte tous les candidats, supprime les doublons, et calcule leur empreinte CLIP."""
    img_cv = pil_to_cv(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 7)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    
    all_fragments = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            # On extrait le patch de l'image originale pour l'embedding
            patch = img_pil.crop((x, y, x + w, y + h))
            all_fragments.append({
                "id": i, "cx": centroids[i][0], "cy": centroids[i][1],
                "area": area, "patch": patch
            })
            
    candidates = suppress_duplicates(all_fragments)
    
    # Calcul des embeddings pour tous les candidats restants
    if candidates:
        patches = [c['patch'] for c in candidates]
        embeddings = model.encode(patches, convert_to_tensor=True, show_progress_bar=False)
        for i, c in enumerate(candidates):
            c['embedding'] = embeddings[i]
            c['is_counted'] = False # Initialisation
            del c['patch'] # On n'a plus besoin de l'image
            
    return candidates
