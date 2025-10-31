# Dog Breed Classifier (120 classes)

A TensorFlow/Keras transfer learning pipeline to classify 120 dog breeds using the Kaggle 'Dog Breed Identification' dataset.

## Dataset layout

Place the dataset at the project root:

```
DogBreedClassfier/
├─ labels.csv              # Kaggle labels (id,breed)
├─ train/                  # Training images: <id>.jpg
│  ├─ 000bec180eb18c7604dcecc8fe0dba07.jpg
│  ├─ ...
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py
# or with flags
python train.py --base_model MobileNetV2 --epochs 15 --batch_size 32 --image_size 224,224
# enable/adjust fine-tuning (second phase)
python train.py --fine_tune true --fine_tune_pct 0.3 --fine_tune_epochs 5 --fine_tune_lr 0.0001
```

- Artifacts are saved under `artifacts/`:
  - `class_distribution.png` – EDA class histogram
  - `best_model.keras` – best checkpoint by val accuracy
  - `final_model.keras` – final model after training
  - `classes.json` – list of class names in model output order
  - `confusion_matrix.png` – optional validation confusion matrix

## Optional: work on a small subset first

The full dataset is large (roughly 0.8–1.2 GB for images). To iterate quickly, create a small stratified subset:

```powershell
python scripts/make_subset.py --per_class 20 --src_dir train --dst_dir train_subset --labels labels.csv --out_labels artifacts/labels_subset.csv
# Then point Config to the subset or override paths via flags
$env:PYTHONPATH = "$PWD"
python train.py --epochs 5
```

Notes:
- Pretrained weights are downloaded on first use (e.g., MobileNetV2 ~9MB, InceptionV3 ~90MB) and cached at `%USERPROFILE%\.keras\models`.
- TensorFlow itself is a large wheel (~300+ MB) but it's a one-time install per environment.
- Fine-tuning unfreezes only the top fraction of the backbone (BatchNorm layers stay frozen) and recompiles with a lower LR for stability.

## Predict top-3 breeds

```powershell
python -m src.predict --image "path\to\dog.jpg" --model artifacts\best_model.keras --classes artifacts\classes.json
```

## Notes

- Default model: MobileNetV2 (ImageNet weights) with frozen base and custom head.
- Images are resized to 224x224 and normalized to [0,1].
- Training uses EarlyStopping and ReduceLROnPlateau.
- You can switch to InceptionV3 by setting `BASE_MODEL_NAME` in `src/config.py` or using a flag in `train.py`.
