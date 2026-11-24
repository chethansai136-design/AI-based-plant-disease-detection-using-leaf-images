<<<<<<< HEAD
# Plant Disease Detection from Leaf Images

An AI-powered plant leaf disease classification project. It uses a fine-tuned Vision Transformer (ViT-B/16) to classify leaf images into 38 PlantVillage disease and healthy-leaf categories. The FastAPI backend returns the predicted class, confidence scores, and (when available) a Grad-CAM visualization to help show which image regions influenced the prediction.
=======
# Machine Learning Pipeline: Vision Transformer (ViT) Plant Disease Diagnosis

This directory contains the methodology, model architecture, training configuration, and explainable AI (XAI) pipeline for the automated foliar disease classification system.
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df

> **Note:** This tool is intended for learning and demonstration. Model predictions are not a substitute for diagnosis by an agricultural expert.

<<<<<<< HEAD
## Features

- Classifies supported leaf images into 38 crop-condition classes.
- Provides the top three predictions and confidence scores.
- Generates a Grad-CAM heatmap for visual model explainability.
- Supports account registration, JWT-based login, and prediction history.
- Includes translated disease names and descriptions in English, Hindi, Kannada, and Telugu.
- Exposes interactive API documentation through FastAPI's Swagger UI.

## Technology

- **Model:** Vision Transformer (ViT-B/16), implemented with PyTorch and TorchVision
- **Explainability:** ViT Grad-CAM
- **API:** FastAPI and Uvicorn
- **Persistence:** SQLite and SQLAlchemy
- **Image processing:** Pillow and NumPy

## Repository structure

```text
.
├── backend/
│   ├── auth/                 # Password hashing, JWT helpers, and request limits
│   ├── database/             # SQLAlchemy models and database connection
│   ├── routes/               # Authentication and prediction endpoints
│   ├── saved_models/         # Put the trained model checkpoint here
│   ├── static/uploads/       # Uploaded images and Grad-CAM outputs
│   ├── classes.json          # Model class labels
│   ├── translations.json     # Disease descriptions and translations
│   ├── main.py               # FastAPI application
│   ├── ml_inference.py       # ViT inference and Grad-CAM
│   └── requirements.txt      # Python dependencies
├── ml/
│   ├── plant_disease_vit_training.ipynb
│   ├── evaluate_samples.py
│   └── README.md             # Training and model details
└── README.md
```
=======
## 1. Model Architecture

The core diagnosis engine leverages a **Vision Transformer (ViT-B/16)** fine-tuned for multi-class plant pathology classification:
- **Base Architecture**: `torchvision.models.vit_b_16`
- **Input Resolution**: $224 \times 224 \times 3$ RGB
- **Patch Size**: $16 \times 16$ (generating a $14 \times 14 = 196$ spatial token grid)
- **Embedding Dimension**: $D = 768$
- **Transformer Encoder**: 12 Multi-Head Self-Attention (MSA) layers, 12 attention heads per layer
- **Classification Head**: `Linear(768, 38)` producing unnormalized logits over 38 botanical pathology classes
- **Inference Latency**: $\approx 120\text{ ms}$ on standard CPU; $< 15\text{ ms}$ on CUDA GPU
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df

## Getting started

<<<<<<< HEAD
### 1. Set up the backend

From the repository root, create and activate a virtual environment, then install the dependencies:

**Windows PowerShell**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
=======
## 2. Dataset Taxonomy & Coverage

The model is trained and validated on the benchmark **PlantVillage** dataset, covering **38 foliar conditions** across 14 crop cultivars:

| Cultivar | Conditions & Pathologies |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Yellow Leaf Curl Virus (TYLCV), Mosaic Virus (ToMV), Healthy |
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df

**macOS / Linux**

<<<<<<< HEAD
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables
=======
## 3. Image Preprocessing & Normalization

All ingested leaves undergo standard ImageNet transfer learning normalization:
$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

1. **Format Handling**: Converted to 3-channel RGB (alpha channel discarded if PNG).
2. **Bilinear Resizing**: Resampled to $224 \times 224$ pixels.
3. **Tensor Conversion**: Normalized from $[0, 255]$ integer space to $[0.0, 1.0]$ float tensor.
4. **Channel Normalization**: Scaled via standard deviation and shifted via channel mean.
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df

From the `backend` directory, copy the example environment file:

<<<<<<< HEAD
```powershell
Copy-Item .env.example .env
```

Set `SECRET_KEY` in `.env` to a unique, private value before using the API beyond local testing. The example database setting uses SQLite:

```env
SECRET_KEY=replace_with_a_long_random_secret
DATABASE_URL=sqlite:///./app.db
```
=======
## 4. Explainable AI: ViT Grad-CAM

To satisfy transparency and clinical validation requirements, the inference service couples predictions with **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
1. Gradients and forward activations are captured from the final LayerNorm block (`encoder.layers[-1].ln_1`).
2. The `[CLS]` token is separated from the 196 spatial patch tokens ($14 \times 14$).
3. Global average pooled gradients weight the spatial token activations:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A_k\right)$$
4. The resulting $14 \times 14$ activation map is normalized and upscaled to full image dimensions using bilinear interpolation.
5. A JET heatmap is alpha-blended ($\alpha = 0.45$) over the original leaf to visually delineate the pathogen lesions.
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df

Do not commit `.env` or share your secret key.

<<<<<<< HEAD
### 3. Add the trained model weights

Place the trained checkpoint at:

```text
backend/saved_models/best_plant_disease_vit.pth
```

Model weights are excluded from Git, so they may not be present after cloning the repository. The API can start without the checkpoint, but its predictions will not be meaningful until the trained weights are supplied.

### 4. Start the API

From the `backend` directory, run:

```bash
uvicorn main:app --reload --port 8000
```

Then open:

- Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

## API overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register an account |
| `POST` | `/auth/login` | Log in and receive a bearer token |
| `POST` | `/predict` | Upload a leaf image and get a disease prediction |
| `GET` | `/predict/history` | Retrieve the signed-in user's prediction history |
| `GET` | `/health` | Check whether the API is running |

Prediction requests require a bearer token and a `.jpg`, `.jpeg`, or `.png` image up to 10 MB. Include the returned access token as a Bearer token in the `Authorization` HTTP header for protected endpoints. The login endpoint accepts form fields `username` and `password`; registration accepts JSON with `username`, `email`, `password`, and optional `preferred_language` (`en`, `hi`, `kn`, or `te`).

A prediction response includes the predicted class, confidence, top-three candidate classes, image and Grad-CAM paths, and a translated disease name and description where available. See `/docs` for the full request and response schemas.

## Model and dataset

The inference pipeline resizes images to 224 × 224 pixels and applies ImageNet normalization before passing them to ViT-B/16. The class mapping in [`backend/classes.json`](backend/classes.json) covers 38 PlantVillage categories across 14 crop types. Grad-CAM highlights image regions that contribute to the selected prediction; it is an explanatory visualization, not proof of a diagnosis.

For training and implementation details, see [`ml/README.md`](ml/README.md) and the [training notebook](ml/plant_disease_vit_training.ipynb).

## Testing

From the `backend` directory, run the included test script:

```bash
python test.py
```

## Acknowledgments

- PlantVillage dataset: Hughes and Salathé, “An open access repository of images on plant health to enable the development of mobile disease diagnostics,” 2015.
- Grad-CAM: Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” ICCV, 2017.
=======
## 5. Model Weights & Deployment

- **Weights File**: `backend/saved_models/best_plant_disease_vit.pth` (343 MB)
- **Online Release**: Available for download under the repository's GitHub Releases tab (`v1.0.0`).
- **Classes Mapping**: [`backend/classes.json`](../backend/classes.json)
- **Agronomic Translations**: [`backend/translations.json`](../backend/translations.json) (EN, HI, KN, TE)
>>>>>>> b10cf27fd08f30b92ae04d05fa3943be000204df
