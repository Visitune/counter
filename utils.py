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
    if num_clusters == 0: return []
    colors = [tuple(map(int, hsv_to_bgr(i * (180.0 / num_clusters), 255, 255))) for i in range(num_clusters)]
    return colors

def hsv_to_bgr(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]

def overlay_clustered_objects(image_pil, objects, selected_cluster_id=None):
    img = pil_to_cv(image_pil.copy())
    if not objects: return cv_to_pil(img)

    cluster_ids = {obj.get('cluster_id') for obj in objects if obj.get('cluster_id') is not None}
    colors = get_cluster_colors(len(cluster_ids))
    cluster_color_map = {cid: color for cid, color in zip(sorted(list(cluster_ids)), colors)}

    for obj in objects:
        cid = obj.get('cluster_id')
        if cid is None: continue

        color = cluster_color_map.get(cid, (0,0,0))
        radius = 10 if cid == selected_cluster_id else 6
        
        if selected_cluster_id is not None and cid != selected_cluster_id:
            color = (180, 180, 180) # Griser les non-sélectionnés
        
        cv2.circle(img, (int(obj['cx']), int(obj['cy'])), radius, color, -1, lineType=cv2.LINE_AA)
        
    return cv_to_pil(img)

def calibrate_hsv_from_click(img_hsv, x, y, tolerance={'h': 15, 's': 70, 'v': 70}):
    x, y = int(x), int(y)
    patch = img_hsv[max(0, y-5):y+5, max(0, x-5):x+5]
    if patch.size == 0: return None
    
    mean_hsv = np.mean(patch, axis=(0, 1))
    h, s, v = mean_hsv
    
    return {
        'h_min': int(max(0, h - tolerance['h'])), 'h_max': int(min(179, h + tolerance['h'])),
        's_min': int(max(0, s - tolerance['s'])), 's_max': int(min(255, s + tolerance['s'])),
        'v_min': int(max(0, v - tolerance['v'])), 'v_max': int(min(255, v + tolerance['v']))
    }

def process_image(img_pil, model, hsv_params, min_area, n_clusters):
    """Fonction unique qui gère la détection, l'embedding et le clustering."""
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
    
    if not candidates:
        return [], mask, img_hsv

    embeddings = model.encode([c['patch'] for c in candidates], convert_to_tensor=True, show_progress_bar=False)
    for i, c in enumerate(candidates):
        c['embedding'] = embeddings[i].cpu().numpy()
        del c['patch']

    if len(candidates) < n_clusters:
        for i, obj in enumerate(candidates): obj['cluster_id'] = i
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(np.array([c['embedding'] for c in candidates]))
        for i, c in enumerate(candidates): c['cluster_id'] = kmeans.labels_[i]
            
    return candidates, mask, img_hsv
