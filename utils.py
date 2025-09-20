# utils.py

import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ================= Conversion et Dessin =================

def pil_to_cv(img_pil):
    """Convertit une image PIL (RGB) en image OpenCV (BGR)."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    """Convertit une image OpenCV (BGR) en image PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_objects(image_pil, objects, radius=6):
    """Dessine des points colorés en fonction du statut de chaque objet."""
    img = pil_to_cv(image_pil.copy())
    status_colors = {
        "confirmed": (0, 255, 0),    # Vert
        "rejected": (0, 0, 255),     # Rouge
        "proposed_good": (150, 255, 150), # Vert clair
        "proposed_bad": (150, 150, 255),  # Rouge clair
        "neutral": (180, 180, 180)   # Gris
    }
    
    for obj in objects:
        cx, cy = int(obj['cx']), int(obj['cy'])
        color = status_colors.get(obj['status'], (255, 255, 0)) # Jaune par défaut
        cv2.circle(img, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        
    return cv_to_pil(img)

def nearest_object_index(objects, qx, qy):
    """Trouve l'index de l'objet le plus proche d'un point de clic."""
    if not objects: return None, float('inf')
    
    # Utilise numpy pour une recherche vectorisée et rapide
    positions = np.array([(obj['cx'], obj['cy']) for obj in objects])
    query_point = np.array([qx, qy])
    distances_sq = np.sum((positions - query_point)**2, axis=1)
    
    kmin = np.argmin(distances_sq)
    dmin = np.sqrt(distances_sq[kmin])
    
    return kmin, dmin

# ================= Détection des candidats et extraction de caractéristiques =================

def extract_features(img_lab, labels_matrix, stats, comp_idx):
    """Extrait un vecteur de caractéristiques pour un seul composant."""
    mask = (labels_matrix == comp_idx)
    
    # 1. Caractéristiques de couleur (moyenne et écart-type sur L, a, b)
    pixels = img_lab[mask]
    mean_color = pixels.mean(axis=0)
    std_color = pixels.std(axis=0)
    
    # 2. Caractéristiques de forme
    area = stats[comp_idx, cv2.CC_STAT_AREA]
    
    # Calcul du périmètre pour la circularité
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours and contours[0].shape[0] > 2 else 0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    # Concatène toutes les caractéristiques en un seul vecteur
    features = np.concatenate([
        mean_color,
        std_color,
        [area],
        [circularity]
    ])
    return features

def detect_candidate_objects(img_pil, min_area=50, max_area=50000):
    """
    Détecte un sur-ensemble de tous les objets potentiels dans l'image
    et extrait leurs caractéristiques pour une classification ultérieure.
    """
    img_cv = pil_to_cv(img_pil)
    img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Utiliser un seuillage adaptatif pour être robuste aux variations d'éclairage
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 61, 7)
    
    # Nettoyage morphologique
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    
    objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            features = extract_features(img_lab, labels, stats, i)
            
            objects.append({
                "id": i,
                "cx": cx,
                "cy": cy,
                "features": features,
                "status": "neutral" # Statuts possibles : neutral, confirmed, rejected, proposed_good, proposed_bad
            })
            
    return objects

# ================= Classification interactive =================

def update_classifications(objects):
    """
    Met à jour les propositions pour tous les objets neutres
    en se basant sur les objets déjà confirmés ou rejetés par l'utilisateur.
    """
    confirmed_objs = [obj for obj in objects if obj['status'] == 'confirmed']
    rejected_objs = [obj for obj in objects if obj['status'] == 'rejected']
    
    # Si pas assez d'exemples, ne rien proposer
    if not confirmed_objs and not rejected_objs:
        return objects

    # Préparer les données pour l'entraînement
    train_features = []
    train_labels = []
    for obj in confirmed_objs:
        train_features.append(obj['features'])
        train_labels.append(1) # 1 pour "bon"
    for obj in rejected_objs:
        train_features.append(obj['features'])
        train_labels.append(0) # 0 pour "mauvais"
        
    # Standardiser les caractéristiques est crucial pour les classifieurs basés sur la distance
    scaler = StandardScaler().fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    
    # Entraîner un classifieur simple et rapide (K-Nearest Neighbors)
    # n_neighbors doit être au max le nombre d'échantillons
    n_neighbors = min(len(train_features), 5)
    if n_neighbors == 0: return objects
    
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_features_scaled, train_labels)

    # Prédire le statut des objets neutres
    for obj in objects:
        if obj['status'] in ["neutral", "proposed_good", "proposed_bad"]:
            features_scaled = scaler.transform([obj['features']])
            prediction = classifier.predict(features_scaled)[0]
            obj['status'] = "proposed_good" if prediction == 1 else "proposed_bad"
            
    return objects
