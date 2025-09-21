# utils.py

import numpy as np
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
def overlay_objects(image_pil, objects, radius=8):
    img = pil_to_cv(image_pil.copy())
    for obj in objects:
        color = (0, 255, 0) if obj.get('is_counted', False) else (180, 180, 180)
        cv2.circle(img, (int(obj['cx']), int(obj['cy'])), radius, color, -1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)
    
def suppress_duplicates(objects, radius_factor=2.0):
    if not objects: return []
    objects.sort(key=lambda o: o['area'], reverse=True)
    kept_objects = []
    # ... (le reste de la fonction est inchangé)
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

# NOUVELLE FONCTION CLÉ
def calibrate_hsv_from_click(img_hsv, x, y, tolerance={'h': 15, 's': 60, 'v': 60}):
    """Calcule la plage HSV optimale à partir d'un clic sur l'image."""
    x, y = int(x), int(y)
    
    # Échantillonner une petite zone autour du clic pour plus de robustesse
    patch = img_hsv[max(0, y-5):y+5, max(0, x-5):x+5]
    mean_hsv = np.mean(patch, axis=(0, 1))
    
    h, s, v = mean_hsv[0], mean_hsv[1], mean_hsv[2]
    
    h_min = max(0, h - tolerance['h'])
    h_max = min(179, h + tolerance['h'])
    s_min = max(0, s - tolerance['s'])
    s_max = min(255, s + tolerance['s'])
    v_min = max(0, v - tolerance['v'])
    v_max = min(255, v + tolerance['v'])
    
    return {'h_min': int(h_min), 'h_max': int(h_max), 's_min': int(s_min), 
            's_max': int(s_max), 'v_min': int(v_min), 'v_max': int(v_max)}

def detect_candidates_by_color(img_pil, model, min_area, hsv_params):
    img_cv = pil_to_cv(img_pil)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_params['h_min'], hsv_params['s_min'], hsv_params['v_min']])
    upper = np.array([hsv_params['h_max'], hsv_params['s_max'], hsv_params['v_max']])
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # ... (le reste de la fonction est inchangé)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_fragments = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                x, y, w, h = cv2.boundingRect(cnt)
                patch = img_pil.crop((x, y, x + w, y + h))
                all_fragments.append({"cx": cx, "cy": cy, "area": area, "patch": patch})
    candidates = suppress_duplicates(all_fragments)
    if candidates:
        embeddings = model.encode([c['patch'] for c in candidates], convert_to_tensor=True, show_progress_bar=False)
        for i, c in enumerate(candidates):
            c['embedding'] = embeddings[i]
            del c['patch']
    return candidates, mask, img_hsv
