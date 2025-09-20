import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io, cv2, json
from datetime import datetime
import uuid

# Génération d'ID unique pour traçabilité
def generate_batch_id():
    return f"CNT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# Utils OpenCV
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def overlay_points(img_pil, points, color=(0,255,0), radius=8):
    img = pil_to_cv(img_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, -1)
        cv2.circle(img, (int(x),int(y)), radius+2, (255,255,255), 2)  # Contour blanc
    return cv_to_pil(img)

# Segmentation par seuillage automatique
def auto_segment_food(img_cv, method='adaptive'):
    """Segmentation automatique spécialisée pour produits alimentaires"""
    
    # Conversion en HSV pour mieux séparer les objets du fond
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    if method == 'adaptive':
        # Seuillage adaptatif sur la composante Value
        gray = hsv[:,:,2]
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESHOLD_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
    elif method == 'kmeans':
        # K-means clustering (plus lourd mais plus précis)
        data = img_cv.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = centers[labels.flatten()].reshape(img_cv.shape).astype(np.uint8)
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Otsu simple
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return thresh

def count_objects_advanced(mask, min_area=100, max_area=10000, min_circularity=0.3):
    """Comptage avancé avec filtres de forme"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    
    valid_objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_area <= area <= max_area:
            # Extraction du contour pour analyse de forme
            object_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                
                # Filtres de forme
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity >= min_circularity:
                        cx, cy = centroids[i]
                        valid_objects.append({
                            'center': (float(cx), float(cy)),
                            'area': int(area),
                            'circularity': float(circularity)
                        })
    
    return valid_objects

# Configuration Streamlit
st.set_page_config(page_title="Comptage Food Safety Pro", layout="wide")

# Header avec branding professionnel
st.markdown("""
# 🔬 Comptage Automatique - Food Safety Pro
### Interface professionnelle pour l'industrie alimentaire
""")

# Sidebar avec métadonnées de traçabilité
with st.sidebar:
    st.header("🏭 Informations Traçabilité")
    
    if 'batch_id' not in st.session_state:
        st.session_state.batch_id = generate_batch_id()
    
    st.code(st.session_state.batch_id, language=None)
    
    operator = st.text_input("Opérateur", placeholder="Nom de l'opérateur")
    production_line = st.selectbox("Ligne de production", ["L1", "L2", "L3", "L4"])
    lot_number = st.text_input("N° Lot", placeholder="LOT202X-XXX")
    supplier = st.text_input("Fournisseur", placeholder="Nom du fournisseur")
    
    st.header("⚙️ Paramètres Technique")
    method = st.selectbox("Méthode segmentation", 
                         ["adaptive", "kmeans", "otsu"],
                         help="Adaptive recommandé pour la plupart des cas")
    min_area = st.slider("Aire min (px²)", 50, 1000, 150)
    max_area = st.slider("Aire max (px²)", 1000, 50000, 8000)
    min_circularity = st.slider("Circularité min", 0.1, 1.0, 0.4, 0.1, 
                               help="0.3-0.8 pour objets alimentaires typiques")

# Zone principale
tab1, tab2, tab3 = st.tabs(["📸 Analyse", "📊 Résultats", "📋 Rapport"])

with tab1:
    uploaded_file = st.file_uploader("Chargez l'image à analyser", 
                                    type=['png', 'jpg', 'jpeg'],
                                    help="Formats supportés: PNG, JPG, JPEG")
    
    if uploaded_file:
        # Traitement de l'image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        # Redimensionnement intelligent
        w, h = original_image.size
        max_display_size = 1000
        if max(w, h) > max_display_size:
            scale = max_display_size / max(w, h)
            display_w, display_h = int(w * scale), int(h * scale)
            display_image = original_image.resize((display_w, display_h), Image.LANCZOS)
        else:
            display_image = original_image
            scale = 1.0
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_image, caption=f"Image source ({w}x{h}px)", use_column_width=True)
        
        with col2:
            st.metric("Résolution", f"{w} × {h}")
            st.metric("Taille fichier", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
            if scale != 1.0:
                st.info(f"Affichage redimensionné (×{scale:.2f})")
        
        # Analyse automatique
        if st.button("🚀 Lancer l'analyse automatique", type="primary"):
            with st.spinner("Analyse en cours..."):
                start_time = datetime.now()
                
                # Segmentation
                img_cv = pil_to_cv(display_image)
                mask = auto_segment_food(img_cv, method)
                
                # Comptage avec analyse de forme
                objects = count_objects_advanced(mask, min_area, max_area, min_circularity)
                
                # Points pour overlay
                points = [obj['center'] for obj in objects]
                
                # Adaptation à l'image originale
                if scale != 1.0:
                    original_points = [(x/scale, y/scale) for x, y in points]
                    final_overlay = overlay_points(original_image, original_points, (0, 255, 0), int(10/scale))
                else:
                    original_points = points
                    final_overlay = overlay_points(original_image, points, (0, 255, 0), 10)
                
                display_overlay = overlay_points(display_image, points, (0, 255, 0), 8)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Sauvegarde résultats
                st.session_state.analysis_results = {
                    'objects': objects,
                    'count': len(objects),
                    'display_overlay': display_overlay,
                    'final_overlay': final_overlay,
                    'mask': mask,
                    'processing_time': processing_time,
                    'parameters': {
                        'method': method,
                        'min_area': min_area,
                        'max_area': max_area,
                        'min_circularity': min_circularity
                    },
                    'metadata': {
                        'batch_id': st.session_state.batch_id,
                        'operator': operator,
                        'line': production_line,
                        'lot': lot_number,
                        'supplier': supplier,
                        'filename': uploaded_file.name,
                        'timestamp': start_time.isoformat()
                    }
                }
                
                st.success(f"✅ Analyse terminée en {processing_time:.2f}s")

with tab2:
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🧮 Objets détectés", results['count'])
        with col2:
            avg_area = np.mean([obj['area'] for obj in results['objects']]) if results['objects'] else 0
            st.metric("📏 Aire moyenne", f"{avg_area:.0f} px²")
        with col3:
            avg_circularity = np.mean([obj['circularity'] for obj in results['objects']]) if results['objects'] else 0
            st.metric("⭕ Circularité moy.", f"{avg_circularity:.2f}")
        with col4:
            st.metric("⏱️ Temps traitement", f"{results['processing_time']:.2f}s")
        
        # Visualisations
        col1, col2 = st.columns(2)
        with col1:
            st.image(results['display_overlay'], caption="Résultat du comptage")
        with col2:
            mask_colored = cv2.applyColorMap(results['mask'], cv2.COLORMAP_VIRIDIS)
            mask_pil = cv_to_pil(mask_colored)
            st.image(mask_pil, caption="Masque de segmentation")
        
        # Correction manuelle
        st.subheader("🔧 Correction manuelle")
        col1, col2, col3 = st.columns(3)
        with col1:
            manual_correction = st.number_input("Ajustement (+/-)", 
                                              min_value=-results['count'], 
                                              max_value=100, 
                                              value=0)
        with col2:
            final_count = results['count'] + manual_correction
            st.metric("Compte final", final_count)
        with col3:
            validation_status = st.selectbox("Statut validation", 
                                           ["En attente", "Validé", "Rejeté"])
        
        # Commentaires
        comments = st.text_area("Observations", 
                               placeholder="Commentaires sur l'analyse...")
        
        # Mise à jour des résultats
        if manual_correction != 0 or comments or validation_status != "En attente":
            st.session_state.analysis_results['final_count'] = final_count
            st.session_state.analysis_results['correction'] = manual_correction
            st.session_state.analysis_results['validation_status'] = validation_status
            st.session_state.analysis_results['comments'] = comments

with tab3:
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        st.subheader("📋 Rapport d'analyse")
        
        # Données du rapport
        report_data = {
            'ID_Analyse': results['metadata']['batch_id'],
            'Timestamp': results['metadata']['timestamp'],
            'Operateur': results['metadata']['operator'] or 'Non renseigné',
            'Ligne_Production': results['metadata']['line'],
            'Numero_Lot': results['metadata']['lot'] or 'Non renseigné',
            'Fournisseur': results['metadata']['supplier'] or 'Non renseigné',
            'Fichier_Image': results['metadata']['filename'],
            'Methode_Segmentation': results['parameters']['method'],
            'Aire_Min_px2': results['parameters']['min_area'],
            'Aire_Max_px2': results['parameters']['max_area'],
            'Circularite_Min': results['parameters']['min_circularity'],
            'Compte_Automatique': results['count'],
            'Correction_Manuelle': results.get('correction', 0),
            'Compte_Final': results.get('final_count', results['count']),
            'Statut_Validation': results.get('validation_status', 'En attente'),
            'Temps_Traitement_s': round(results['processing_time'], 3),
            'Observations': results.get('comments', '')
        }
        
        # Affichage du rapport
        df_report = pd.DataFrame([report_data]).T
        df_report.columns = ['Valeur']
        st.dataframe(df_report, use_container_width=True)
        
        # Export
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export image
            img_buffer = io.BytesIO()
            results['final_overlay'].save(img_buffer, format='PNG')
            st.download_button(
                "📸 Image annotée",
                data=img_buffer.getvalue(),
                file_name=f"{results['metadata']['batch_id']}_resultat.png",
                mime="image/png"
            )
        
        with col2:
            # Export CSV
            df_csv = pd.DataFrame([report_data])
            csv_data = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Rapport CSV",
                data=csv_data,
                file_name=f"{results['metadata']['batch_id']}_rapport.csv",
                mime="text/csv"
            )
        
        with col3:
            # Export JSON (données complètes)
            json_data = json.dumps(results, default=str, indent=2, ensure_ascii=False)
            st.download_button(
                "🔧 Données JSON",
                data=json_data.encode('utf-8'),
                file_name=f"{results['metadata']['batch_id']}_donnees.json",
                mime="application/json"
            )
    
    else:
        st.info("Effectuez d'abord une analyse pour générer un rapport.")

# Footer
st.markdown("---")
st.caption("🔬 Comptage Food Safety Pro - Version 1.0 | Conforme aux standards de traçabilité alimentaire")
