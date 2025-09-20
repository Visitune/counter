# utils.py

import numpy as np
import cv2
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# ================= Conversion et Dessin =================

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_objects(image_pil, objects, radius=6):
    """Dessine les points avec des couleurs indiquant leur statut."""
    img = pil_to_cv(image_pil.copy())
    # Statut -> Couleur (BGR)
    status_colors = {
        "confirmed": (0, 255, 0),       # Vert vif : Confirmé par l'utilisateur
        "rejected": (0, 0, 255),        # Rouge vif : Rejeté par l'utilisateur
        "proposed_good": (120, 255, 120),# Vert clair : Proposé par l'IA comme bon
        "proposed_bad": (120, 120, 255), # Rose : Proposé par l'IA comme mauvais
        "neutral": (180, 180, 180)      # Gris : Non étiqueté
    }
    
    for obj in objects:
        cx, cy = int(obj['cx']), int(obj['cy'])
        color = status_colors.get(obj['status'], (255, 255, 0)) # Jaune si statut inconnu
        cv2.circle(img, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        
    return cv_to_pil(img)

def nearest_object_index(objects, qx, qy):
    """Trouve l'index de l'objet le plus proche d'un point de clic, de manière optimisée."""
    if not objects: return None, float('inf')
    positions = np.array([(obj['cx'], obj['cy']) for obj in objects])
    distances_sq = np.sum((positions - np.array([qx, qy]))**2, axis=1)
    kmin = np.argmin(distances_sq)
    return kmin, np.sqrt(distances_sq[kmin])

# ================= Détection & Caractéristiques =================

def extract_features(img_lab, labels_matrix, stats, comp_idx):
    """Extrait un vecteur de caractéristiques pour un objet (couleur, forme)."""
    mask = (labels_matrix == comp_idx)
    pixels = img_lab[mask]
    
    # Caractéristiques de couleur (plus robustes)
    mean_color = pixels.mean(axis=0)
    std_color = pixels.std(axis=0)
    
    # Caractéristiques de forme
    area = stats[comp_idx, cv2.CC_STAT_AREA]
    x, y, w, h = stats[comp_idx, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
    aspect_ratio = w / h if h > 0 else 1
    
    # Concaténation
    features = np.concatenate([mean_color, std_color, [area], [aspect_ratio]])
    return features

def detect_candidate_objects(img_pil, min_area=50, max_area=50000):
    """Détecte tous les objets potentiels et extrait leurs caractéristiques."""
    img_cv = pil_to_cv(img_pil)
    img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 61, 7)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    
    objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            features = extract_features(img_lab, labels, stats, i)
            objects.append({
                "id": i,
                "cx": centroids[i][0], "cy": centroids[i][1],
                "features": features,
                "status": "neutral"
            })
    return objects

# ================= Apprentissage et Classification =================

def train_and_predict(objects):
    """
    Entraîne un classifieur SVM sur les objets étiquetés par l'utilisateur
    et prédit le statut de tous les autres objets.
    """
    train_features = []
    train_labels = []
    
    for obj in objects:
        if obj['status'] in ['confirmed', 'rejected']:
            train_features.append(obj['features'])
            train_labels.append(1 if obj['status'] == 'confirmed' else 0)

    # Vérifier qu'on a assez de données des deux classes pour apprendre
    if len(set(train_labels)) < 2:
        return objects, "Veuillez confirmer au moins un 'bon' et un 'mauvais' objet."

    # Standardiser les données est crucial pour les SVM
    scaler = StandardScaler().fit(train_features)
    train_features_scaled = scaler.transform(train_features)

    # Entraîner le classifieur SVM
    classifier = SVC(kernel='rbf', probability=True, C=10, gamma='auto')
    classifier.fit(train_features_scaled, train_labels)

    # Prédire sur les objets non-étiquetés
    for obj in objects:
        if obj['status'] not in ['confirmed', 'rejected']:
            features_scaled = scaler.transform([obj['features']])
            prediction = classifier.predict(features_scaled)[0]
            obj['status'] = "proposed_good" if prediction == 1 else "proposed_bad"
            
    return objects, f"Apprentissage terminé. {len(objects)} objets classifiés."
