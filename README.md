# EYE Médicale

Projet réalisé en groupe à l’ESIEE Paris en 2025. On a associé CLIP à Gemma pour générer des descriptions d’images médicales, avec une API FastAPI pour communiquer avec l’interface web.

Avec Lorenzo, j’ai travaillé sur le modèle, la préparation des données, le fine-tuning LoRA et les tests. Youssouf et Guillaume se sont occupés du backend, et Alexandre du frontend.

## Le code

- [`notebooks/entrainement.ipynb`](notebooks/entrainement.ipynb) : images ROCOv2, entraînement, validation et inférence.
- [`src/api.py`](src/api.py) : chargement du modèle et route `/predict`.
- [`src/text_utils.py`](src/text_utils.py) : nettoyage des réponses répétitives.
- [`docs/architecture.md`](docs/architecture.md) : fonctionnement du modèle.

CLIP encode l’image. Une projection linéaire adapte ce vecteur à la taille des embeddings de Gemma et remplace le token image dans le prompt. Gemma produit la description, avec des adaptateurs LoRA pour le fine-tuning.

## Entraînement

Le notebook se lance dans un environnement Python avec un GPU CUDA compatible avec Unsloth. La première cellule installe les bibliothèques utilisées. Les images viennent de [`eltorio/ROCOv2-radiology`](https://huggingface.co/datasets/eltorio/ROCOv2-radiology).

La dernière cellule exporte les adaptateurs LoRA, le tokenizer, la projection visuelle et la configuration dans `model/eye-medicale`. Garder ces fichiers ensemble : l’API utilise la projection entraînée avec le modèle.

## API

Installer [`requirements.txt`](requirements.txt) dans l’environnement GPU. Placer l’export du notebook dans `model/eye-medicale`, puis lancer depuis la racine du dépôt :

```sh
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

La documentation interactive est sur <http://127.0.0.1:8000/docs>. Pour utiliser un autre dossier de poids, définir `EYE_MODEL_DIR`.

```sh
curl -X POST http://127.0.0.1:8000/predict -F "file=@image.png"
```

La réponse est un objet JSON avec un champ `result`.

## Vérifications

```sh
python -m unittest discover -s tests -v
```

Ces tests vérifient le nettoyage du texte, sans charger le modèle. L’entraînement et l’inférence GPU n’ont pas été relancés pour cette mise en ligne. Les poids et les images ne sont pas inclus dans le dépôt.

C’est un prototype étudiant, sans validation clinique. Les réponses peuvent être incorrectes : ce n’est pas un outil de diagnostic.

## Équipe et références

Samuel Zerah et Lorenzo Panza : modèle IA. Youssouf Timite et Guillaume Tricot : backend. Alexandre Coulibaly : frontend.

Le code reprend le notebook d’entraînement de juin 2025 et le backend du projet, avec les adaptations pour l’export et le chargement des poids. Le périmètre de cette version est le modèle et l’API.

[CLIP](https://github.com/openai/CLIP) · [Gemma](https://huggingface.co/unsloth/gemma-2b-bnb-4bit) · [Unsloth](https://github.com/unslothai/unsloth) · [PEFT](https://github.com/huggingface/peft) · [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)

Les modèles, bibliothèques et données gardent leurs licences respectives.
