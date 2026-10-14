# PlantCare AI - Dataset Guide & Setup Instructions

This document provides instructions on legally acquiring, organizing, and preparing the plant disease dataset for model training.

---

## 1. Where to Obtain the PlantVillage Dataset Legally

The **PlantVillage** dataset is a public agricultural benchmark containing over 54,000 leaf images across 38 crop and disease categories, released under the Creative Commons license for research and educational purposes.

### Official & Legal Download Sources:
1. **Kaggle (Recommended & Fastest):**
   - Direct link: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) or [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Free download with a Kaggle account (ZIP archive).
2. **Mendeley Data:**
   - [PlantVillage Open Access Repository](https://data.mendeley.com/datasets/tywbtsjrjv/1)
3. **Official Research Paper Reference:**
   - Hughes, D., & Salathé, M. (2015). *"An open access repository of images on plant health to enable the development of mobile disease diagnostics."* arXiv:1511.08060.

---

## 2. Recommended Directory Structure

To train `train_model.py`, organize your dataset with one subfolder per class:

```text
dataset/
└── PlantVillage/
    ├── Apple___Apple_scab/
    │   ├── image_0001.JPG
    │   └── image_0002.JPG
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Corn_(maize)___Common_rust_/
    ├── Corn_(maize)___Northern_Leaf_Blight/
    ├── Corn_(maize)___healthy/
    ├── Grape___Black_rot/
    ├── Grape___healthy/
    ├── Pepper__bell___Bacterial_spot/
    ├── Pepper__bell___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    ├── Tomato___Leaf_Mold/
    ├── Tomato___Septoria_leaf_spot/
    ├── Tomato___Yellow_Leaf_Curl_Virus/
    └── Tomato___healthy/
```

> **Note on Naming Convention:**
> Folder names use the format `Plant___Disease` separated by triple underscores (`___`).
> The training script dynamically infers class names from these folder names and generates `backend/models/class_indices.json`.

---

## 3. Dataset Configuration & Splitting

When you run `python backend/train_model.py`:
1. **70% Training Set:** Used to optimize model weights. Augmented on-the-fly with random flips, rotations, zooms, and contrast shifts.
2. **15% Validation Set:** Evaluated at the end of each epoch to trigger `EarlyStopping` and `ModelCheckpoint`.
3. **15% Held-Out Test Set:** Evaluated strictly once after training finishes to compute unbiased test accuracy, precision, recall, F1 scores, and the confusion matrix.

This strict partition ensures **zero data leakage**.

---

## 4. Laptop Training Optimization (Low RAM & CPU)

If you are training on a laptop without an NVIDIA GPU or with limited RAM (8GB or 16GB):

### A. Reduce Batch Size
Default batch size is `32`. If you see an `Out of Memory` (OOM) error, lower it to `16`:
```powershell
python backend/train_model.py --batch_size 16 --epochs 10
```

### B. Train on a Subset of Classes (Quick Hackathon Prototype)
Instead of all 38 classes, copy 4–6 folders (e.g. only `Tomato` classes) into `dataset/PlantVillage/`. Training will complete in 10–15 minutes on a standard CPU!

### C. Preprocessing Image Size
Images are automatically resized to `224x224` pixels during loading.

---

## 5. Laboratory Dataset vs. Real-World Field Performance

| Factor | Laboratory Images (PlantVillage) | Real-World Field Conditions |
|---|---|---|
| **Background** | Uniform grey or black paper sheet | Soil, mulch, weeds, hands, tools |
| **Lighting** | Controlled, diffused indoor illumination | Harsh midday sun, direct glare, deep shade |
| **Foliage State** | Single excised leaf flat on a surface | Overlapping leaf clusters, fluttering in wind |
| **Pathology** | Single isolated disease symptom | Co-infections, nutrient deficiencies, insect feeding |

### Mitigations Built Into PlantCare AI:
- **Low-Confidence Warning:** Scans with $<60\%$ confidence prompt the user to retake the photo with better lighting or contact an agronomist.
- **Safety Disclaimer:** The UI prominently emphasizes that computer vision is an advisory aid, not a certified agronomic diagnosis.
- **Organic Care Advice:** Guidance prioritizes safe cultural sanitation and non-chemical methods to prevent improper chemical applications.

---

## 6. Adding New Crop Species or Disease Classes

To add a new crop (e.g., `Strawberry___Leaf_scorch`):
1. Create `dataset/PlantVillage/Strawberry___Leaf_scorch/` and place leaf photos inside.
2. Add the corresponding agronomic profile to `DISEASE_KNOWLEDGE_BASE` in `backend/recommendations.py`.
3. Re-run `python backend/train_model.py`. The model and `class_indices.json` will update automatically!
