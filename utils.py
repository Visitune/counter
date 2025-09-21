# utils.py

import numpy as np
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_clip_model():
    """Charge le modèle d'IA une seule fois."""
    return SentenceTransformer('clip-ViT-B-32')

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_objects(image_pil, objects, positive_indices):
    """Affiche les objets : 'comptés', 'exemples' et 'autres'."""
    img = pil_to_cv(image_pil.copy())
    if not objects: return cv_to_pil(img)

    for i, obj in enumerate(objects):
        is_positive = i in positive_indices
        is_counted = obj.get('is_counted', False)
        
        # Définir couleur et apparence
        if is_positive:
            color = (0, 255, 0) # Vert Vif pour les exemples
            radius = 10
            thickness = 3 # Juste un cercle
        elif is_counted:
            color = (120, 255, 120) # Vert clair pour les objets comptés
            radius = 7
            thickness = -1 # Point rempli
        else:
            color = (180, 180, 180) # Gris pour les autres
            radius = 5
            thickness = -1
            
        cv2.circle(img, (int(obj['cx']), int(obj['cy'])), radius, color, thickness, lineType=cv2.LINE_AA)
            
    return cv_to_pil(img)


def detect_and_embed_candidates(img_pil, model, min_area=50):
    """
    Détecte tous les candidats possibles de manière très large
    et calcule leur empreinte numérique.
    """
    img_cv = pil_to_cv(img_pil)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Paramètres de détection très larges pour ne rien manquer
    lower = np.array([0, 50, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    
    # Nettoyage minimal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                x, y, w, h = cv2.boundingRect(cnt)
                patch = img_pil.crop((x, y, x + w, y + h))
                candidates.append({"cx": cx, "cy": cy, "area": area, "patch": patch})

    if not candidates:
        return []

    # Calculer les embeddings (la partie la plus longue)
    embeddings = model.encode([c['patch'] for c in candidates], convert_to_tensor=False, show_progress_bar=True)
    for i, c in enumerate(candidates):
        c['embedding'] = embeddings[i]
        del c['patch'] # Plus besoin de l'image du patch
            
    return candidates
