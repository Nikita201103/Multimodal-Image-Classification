# 🧠 Multimodal Image Classification using Traditional Feature Extraction

### Author: Nikita Agre
---

## 📘 Project Overview
This project implements a **Multimodal Machine Learning (MML)** pipeline for image classification using both visual and textual data.  
The aim is to compare **unimodal** (image-only and text-only) and **multimodal** (image + text) models using traditional feature extraction techniques.

---

## 🧩 Dataset
**Dataset Used:** MS COCO 2017 Validation Set (≈5,000 images)

- **Images:** 5,000 unique images  
- **Captions:** 25,014 total (≈5 captions per image)  
- **Classes:** 80 object categories  
- **Total Samples:** 24,774 labeled + 240 unlabeled samples  

---

## ⚙️ Feature Extraction

### 🖼️ Image Modality
**Technique:** Canny Edge Detection (`cv2.Canny`)  
Steps:
1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Hysteresis thresholding

**Output:** 4096-dimensional vector per image.

### 📝 Text Modality
**Technique:** Word2Vec (Google News 300 model)

Steps:
1. Lowercasing  
2. Punctuation removal  
3. Tokenization (`nltk.word_tokenize`)  
4. Average Word2Vec embeddings per caption  

**Output:** 300-dimensional vector per caption.

---

## 🔗 Feature Fusion

### 1. Concatenation  
Combined `[Canny + Word2Vec]` = 4396-dimensional vector.

### 2. Element-wise Addition  
Zero-padded text embeddings to match 4096-dim image vector → fused 4096-dim feature vector.

---

## 🧠 Model Architecture

**Model:** Multilayer Perceptron (MLP)

| Setting | Value |
|----------|--------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Binary Cross-Entropy |
| Metrics | Precision, Recall, AUC, Binary Accuracy |
| Epochs | 30 |
| Batch Size | 64 |

**Data Split:** 70% train / 10% validation / 20% test (split by image).

---

## 📊 Results

| Model Type | Fusion | Test AUC | Precision | Recall | Accuracy |
|-------------|---------|-----------|-----------|---------|-----------|
| Multimodal | Concatenation | 0.5884 | 0.5110 | 0.1331 | 0.9618 |
| Multimodal | Addition | 0.5818 | 0.5376 | 0.1211 | 0.9629 |
| Unimodal | Image Only | 0.5889 | 0.4953 | 0.1224 | 0.9622 |
| Unimodal | Text Only | **0.9446** | **0.7986** | **0.5354** | **0.9773** |

📈 **Observation:**  
- Word2Vec (text) features outperformed all others.  
- Canny edge features were too weak for meaningful classification.  
- Concatenation and addition fusion did not improve performance over unimodal text-only models.  

---

## 💬 Discussion
- **Strengths:**  
  - Implemented and compared two early fusion strategies.  
  - Demonstrated Word2Vec’s strong discriminative power.

- **Limitations:**  
  - Weak visual features (Canny).  
  - Simple MLP couldn’t capture complex modality interactions.  
  - Class imbalance inflated accuracy.

---

## 🚀 Future Improvements
- Use CNNs (e.g., ResNet, VGG) for richer image features.  
- Use non-linear fusion or attention-based multimodal fusion.  
- Handle class imbalance via sampling or focal loss.  
- Experiment with transformer-based encoders for text.

---

## 🧰 Requirements

```bash
python>=3.8
tensorflow
keras
opencv-python
gensim
nltk
numpy
pandas
matplotlib
