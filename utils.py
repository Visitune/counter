# utils.py

import numpy as np
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st
from sklearn.cluster import KMeans

@st.cache_resource
def get_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def get_cluster_colors(num_clusters):
    """Génère une liste de couleurs distinctes pour visualiser les clusters."""
    if num_clusters == 0:
        return []
    colors = []
    # Utiliser une roue chromatique HSV pour des couleurs bien distinctes
    for i in range(num_clusters):
        hue = int(i * (180.0 / num_clusters))
        colors.append((hue, 255, 255))
    
    # Convertir les couleurs HSV en BGR pour OpenCV
    bgr_colors = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2BGR)[0][0] for c in colors]
    return [tuple(map(int, color)) for color in bgr_colors]

def overlay_clustered_objects(image_pil, objects, selected_cluster_id=None):
    """Affiche les objets avec une couleur par cluster."""
    img = pil_to_cv(image_pil.copy())
    
    cluster_ids = {obj.get('cluster_id') for obj in objects if obj.get('cluster_id') is not None}
    if not cluster_ids:
        return cv_to_pil(img)
        
    colors = get_cluster_colors(len(cluster_ids))
    cluster_color_map = {cid: color for cid, color in zip(sorted(list(cluster_ids)), colors)}

    for obj in objects:
        cid = obj.get('cluster_id')
        if cid is None: continue

        color = cluster_color_map.get(cid)
        radius = 10 if cid == selected_cluster_id else 6
        thickness = -1 # Rempli
        
        # Mettre en évidence le cluster sélectionné
        if selected_cluster_id is not None and cid != selected_cluster_id:
            color = (128, 128, 128) # Griser les autres
        
        cv2.circle(img, (int(obj['cx']), int(obj['cy'])), radius, color, thickness, lineType=cv2.LINE_AA)
        
    return cv_to_pil(img)
    
# ===================================================================
# LA FONCTION QUI MANQUAIT EST MAINTENANT INCLUSE CI-DESSOUS
# ===================================================================
def calibrate_hsv_from_click(img_hsv, x, y, tolerance={'h': 15, 's': 60, 'v': 60}):
    """Calcule la plage HSV optimale à partir d'un clic sur l'image."""
    x, y = int(x), int(y)
    
    # Échantillonner une petite zone autour du clic pour plus de robustesse
    patch = img_hsv[max(0, y-5):y+5, max(0, x-5):x+5]
    if patch.size == 0:
        return None # Clic en dehors des limites
        
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

def cluster_objects(objects, n_clusters=5):
    """Regroupe les objets en clusters en fonction de leur embedding."""
    if not objects or len(objects) < n_clusters:
        for i, obj in enumerate(objects):
            obj['cluster_id'] = i
        return objects

    embeddings = np.array([obj['embedding'].cpu() for obj in objects])
    
    # Utiliser KMeans pour un clustering rapide
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    
    for i, obj in enumerate(objects):
        obj['cluster_id'] = kmeans.labels_[i]
        
    return objects

def detect_and_embed(img_pil, model, hsv_params, min_area=150):
    """Combine la détection par couleur et l'extraction des embeddings."""
    img_cv = pil_to_cv(img_pil)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_params['h_min'], hsv_params['s_min'], hsv_params['v_min']])
    upper = np.array([hsv_params['h_max'], hsv_params['s_max'], hsv_params['v_max']])
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
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
    
    if candidates:
        embeddings = model.encode([c['patch'] for c in candidates], convert_to_tensor=True, show_progress_bar=False)
        for i, c in enumerate(candidates):
            c['embedding'] = embeddings[i]
            del c['patch']
            
    return candidates
