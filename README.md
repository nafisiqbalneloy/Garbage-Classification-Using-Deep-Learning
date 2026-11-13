# ♻️ Garbage Classification with EfficientNet-B6

An image classifier that sorts waste into **10 categories** using transfer learning. A fine-tuned **EfficientNet-B6** (PyTorch) reaches **94.18% accuracy on a held-out test set** and is served through a **Gradio** web app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-torchvision-ee4c2c)
![Gradio](https://img.shields.io/badge/Demo-Gradio-orange)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.18%25-brightgreen)

---

## 📌 Overview

Correct waste sorting is the first step of effective recycling. This project fine-tunes an ImageNet-pretrained EfficientNet-B6 to recognise ten waste types from a photo, and wraps it in a simple web interface where anyone can upload an image and see class probabilities.

**Pipeline:** data download → EDA → cleaning → stratified split → training → fine-tuning → evaluation → Gradio deployment.

## 📂 Dataset

[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (Kaggle, MIT license). The `original/` folder is used: **12,259 images** across 10 classes.

| Class | Images | Class | Images |
|---|---:|---|---:|
| clothes | 1,892 | paper | 1,336 |
| glass | 1,736 | metal | 930 |
| plastic | 1,597 | battery | 756 |
| shoes | 1,449 | biological | 699 |
| cardboard | 1,411 | trash | 453 |

The dataset is **class-imbalanced** (`clothes` has roughly 4× the images of `trash`), so the train/val/test split is stratified.

## 🧪 Methodology

**Preprocessing and augmentation**
- Removed non-image and corrupted files.
- Resize to **224×224**, normalize with ImageNet mean/std.
- Training augmentation: horizontal and vertical flips, random rotation (±30°), color jitter (brightness and contrast 0.2).

**Stratified split (`random_state=42`)**

| Train | Validation | Test |
|---:|---:|---:|
| 8,581 (70%) | 1,839 (15%) | 1,839 (15%) |

**Model:** `torchvision` EfficientNet-B6 with ImageNet (`IMAGENET1K_V1`) weights; the final classifier layer is replaced with a 10-way linear layer.

**Training**
1. **Stage 1:** 3 epochs, Adam, lr = 1e-3, cross-entropy loss, batch size 16.
2. **Stage 2 (fine-tuning):** 2 more epochs at a lower lr = 1e-4.

Trained on a Tesla T4 GPU in Google Colab.

## 📊 Results

| Stage | Train Acc. | Validation Acc. |
|---|---:|---:|
| After stage 1 (3 epochs) | 88.63% | 86.73% |
| After fine-tuning (+2 epochs) | 94.34% | **93.20%** |

| Final evaluation | Accuracy |
|---|---:|
| **Held-out test set (1,839 images)** | **94.18%** |

Lowering the learning rate for a short second stage lifted validation accuracy by about 6.5 points, and the test score matches the validation score, so the model is not overfitting to the validation set.

**Confusion matrix observations (test set)**
- Strongest classes: `clothes` (281 of 284 correct), `shoes`, `cardboard`, `battery`.
- Most common confusions are between materials that look alike: `glass` ↔ `metal` / `plastic`, and `plastic` ↔ `metal`.
- `trash` is the weakest class (56 of 68 correct), consistent with it having the fewest training images and being a catch-all category.

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/garbage-classification.git
cd garbage-classification
pip install torch torchvision gradio pillow scikit-learn matplotlib seaborn pandas kaggle
```

### 2. Download the dataset

Put your Kaggle API token at `~/.kaggle/kaggle.json`, then:

```bash
kaggle datasets download sumn2u/garbage-classification-v2
unzip garbage-classification-v2.zip -d garbage-classification
```

### 3. Train and evaluate

Open `Garbage_Classification.ipynb` (written for Google Colab with a GPU) and run all cells. Adjust `data_dir` if your dataset lives elsewhere. The notebook saves `garbage_eff_b6.pth` after stage 1 and `garbage_eff_b6_finetuned.pth` after fine-tuning.

### 4. Launch the web app

The last cells of the notebook start the Gradio demo. To run it on its own, with the fine-tuned weights in the same folder:

```python
import torch, torch.nn as nn
from torchvision import models, transforms
import gradio as gr

class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b6(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('garbage_eff_b6_finetuned.pth', map_location=device))
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img):
    x = tf(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    return {c: float(p[i]) for i, c in enumerate(class_names)}

gr.Interface(predict, gr.Image(type='pil'), gr.Label(num_top_classes=5),
             title='Garbage Classification System').launch()
```

Class order is alphabetical, matching `ImageFolder`.

## 🗂️ Project Structure

```
.
├── Garbage_Classification.ipynb   # EDA, training, evaluation, deployment
└── README.md
```

> Model weights (`.pth`) are large and are not committed. Host them with [Git LFS](https://git-lfs.com/), GitHub Releases or Hugging Face.

## 🔭 Future Work

- Add a learning-rate scheduler, early stopping and more epochs.
- Handle class imbalance (weighted loss or oversampling), especially for `trash`.
- Report per-class precision, recall and F1.
- Compare with other backbones (e.g. EfficientNet-B2, MobileNet) for a lighter deployment.
- Export to ONNX or TorchScript for mobile and edge use.

## 📄 License

Add your preferred license for the code (e.g. MIT). The dataset is MIT-licensed, via [sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2).
