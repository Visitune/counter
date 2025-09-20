# Comptage agrée — Basic (OpenCV, Streamlit Cloud ready)

**Objectif** : app Streamlit légère et 100% compatible Streamlit Cloud pour le comptage basique (crevettes, riz, graines, lentilles), sans GPU ni poids IA.

## Déploiement Streamlit Cloud
1. Pousse ce dossier sur **GitHub** (les fichiers, pas le zip).
2. Dans Streamlit Cloud : **New app** → repo → `app.py`.
3. Dans l'UI de déploiement, onglet **Advanced settings** → choisis **Python 3.12**.
4. Laisse s’installer `packages.txt` (libgl1, libglib2.0-0) et les `requirements`.
