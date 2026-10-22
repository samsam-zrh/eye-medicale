# EYE Médicale

Projet de groupe à l’ESIEE Paris, en 2025. On a utilisé CLIP et Gemma pour générer des descriptions d’images médicales, avec une API FastAPI.

J’ai travaillé sur le modèle et le fine-tuning LoRA avec Lorenzo Panza. Youssouf Timite et Guillaume Tricot ont fait le backend, Alexandre Coulibaly le frontend.

## Lancer le projet

Le [notebook](notebooks/entrainement.ipynb) contient l’entraînement sur [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology). Il faut un GPU CUDA compatible avec Unsloth. Sa dernière cellule exporte les poids, le tokenizer et la projection visuelle dans `model/eye-medicale` : garder ces fichiers ensemble.

Depuis la racine, dans l’environnement GPU :

```sh
pip install -r requirements.txt
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

Ouvrir <http://127.0.0.1:8000/docs> pour envoyer une image à `/predict`. Le dossier du modèle peut être changé avec `EYE_MODEL_DIR`.

Le détail du modèle est dans [architecture.md](docs/architecture.md).

## Tests

```sh
python -m unittest discover -s tests -v
```

Ils couvrent le nettoyage du texte. L’entraînement et l’inférence GPU n’ont pas été relancés pour cette version. Les poids et les images ne sont pas inclus.

Prototype étudiant sans validation clinique, pas un outil de diagnostic.

[CLIP](https://github.com/openai/CLIP) · [Gemma](https://huggingface.co/unsloth/gemma-2b-bnb-4bit) · [Unsloth](https://github.com/unslothai/unsloth) · [PEFT](https://github.com/huggingface/peft). Les modèles, bibliothèques et données gardent leurs licences.
