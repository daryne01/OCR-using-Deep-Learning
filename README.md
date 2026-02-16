# Amazigh Handwriting Recognition — Deep Learning Model Comparison

> MSc Data Science Dissertation Project — Enhancing Handwritten Character Recognition for Berber Scripts Using Deep Learning Techniques

## The Problem

Berber (Amazigh) languages are spoken by millions of people across North Africa, yet there are virtually no reliable OCR systems for their writing systems. Both **Tifinagh** (the traditional script) and **Latin Amazigh** (the Latin-based alphabet) are underrepresented in digital tools. This means Berber text can't be easily digitized, searched, or processed — limiting the language's presence in modern technology and threatening its preservation.

## The Solution

This project tackles the problem head-on by building and comparing **6 different deep learning architectures** for handwritten character recognition across both Berber scripts. Rather than just picking one model and hoping for the best, I systematically evaluated which approaches work best for these specific scripts — giving concrete recommendations for building better Amazigh OCR systems.

### Models Compared

| Model | Tifinagh Accuracy | Latin Amazigh Accuracy | Type |
|-------|:-:|:-:|------|
| **VGG-Like CNN** | **98.8%** | **98.9%** | CNN |
| **ResNet (Residual Network)** | **98.8%** | 98.6% | CNN |
| EfficientNetB0 | 95.9% | 97.2% | Transfer Learning |
| LeNet-5 | 97.6% | 97.8% | Classic CNN |
| MLP (Multi-Layer Perceptron) | 94.7% | 96.8% | Fully Connected |
| Capsule Network | 3% (failed) | 3% (failed) | Capsule Network |

**Key finding:** The VGG-like CNN and ResNet consistently outperformed all other architectures, achieving near-99% accuracy on both scripts. The Capsule Network failed to converge (NaN loss), which itself is a valuable finding — not every architecture suits every problem.

## Dataset

**Berber-MNIST** — A handwritten character dataset containing:
- **33 character classes** for each script
- **Tifinagh** (traditional Berber script — ⵜⵉⴼⵉⵏⴰⵖ)
- **Latin Amazigh** (Latin-based Berber alphabet)
- Images resized to 28×28 grayscale with quality filtering (low-contrast images removed)

## Project Structure

```
├── Berber_dataset_preprocessing_and_analysis.ipynb   # Data exploration, cleaning, visualization
├── Amazigh-DeeperExploration.ipynb                   # Latin Amazigh — 6 model comparison
├── tifinagh-DeeperExploration.ipynb                  # Tifinagh — 6 model comparison
├── OCR System.ipynb                                  # End-to-end OCR pipeline using Tesseract
├── Project_Proposal-23012721.docx                    # Original MSc project proposal
└── README.md
```

## Technical Approach

### Data Pipeline
- Grayscale conversion and 28×28 resizing
- Quality control filtering (standard deviation > 0.05 to remove blank/low-contrast images)
- 80/20 train-test split with data augmentation (rotation, shift, shear, zoom)

### Architectures Implemented

**Capsule Network** — Custom implementation with squash activation, primary capsules, digit capsules, and a decoder network for reconstruction. Explored whether capsule networks' ability to preserve spatial hierarchies would benefit script recognition (finding: they didn't converge on this dataset).

**EfficientNetB0** — Transfer learning approach adapted for small grayscale character images. Trained from scratch (no pretrained weights) to handle the domain shift from natural images to handwritten characters.

**LeNet-5** — Classic CNN architecture (LeCun et al., 1998) using tanh activations and average pooling. Served as a strong baseline despite being one of the oldest architectures tested.

**MLP** — Simple fully-connected network (Flatten → 128 → 64 → softmax) to establish a floor for comparison. Surprisingly competitive at ~95% accuracy.

**VGG-Like CNN** — Small VGG-inspired network with stacked 3×3 convolutions, max pooling, and a dense classification head. Top performer across both scripts.

**ResNet (Residual Network)** — Custom residual blocks with skip connections and 1×1 convolution shortcuts. Matched VGG performance while being more theoretically scalable.

### OCR System
Built an end-to-end OCR pipeline using **Tesseract** with OpenCV preprocessing (thresholding, noise removal, dilation) to demonstrate practical application of the character recognition models.

## Tech Stack

- **Python 3.11** / **Jupyter Notebook**
- **TensorFlow / Keras** — Model building, training, evaluation
- **scikit-learn** — Data splitting, label encoding, metrics
- **OpenCV** — Image preprocessing for OCR pipeline
- **Tesseract (pytesseract)** — OCR engine integration
- **NumPy / Pandas / Matplotlib** — Data manipulation and visualization
- **PIL (Pillow)** — Image loading and processing

## What I Learned

- **Architecture matters more than complexity.** The simple VGG-like network outperformed the more complex Capsule Network and EfficientNetB0. For structured, small-image classification tasks, straightforward CNNs with stacked convolutions remain hard to beat.
- **Failure is data.** The Capsule Network producing NaN loss across all epochs taught me about numerical instability in routing-by-agreement and the importance of gradient monitoring.
- **Data quality > data quantity.** Filtering out low-contrast images with a simple standard deviation threshold improved model performance across the board.
- **Domain-specific challenges exist.** Some Tifinagh characters are visually similar (like ⵜ and ⵟ), which presents unique challenges that general-purpose OCR systems don't handle well.

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/amazigh-handwriting-recognition.git
cd amazigh-handwriting-recognition

# Install dependencies
pip install tensorflow scikit-learn pillow matplotlib numpy pandas opencv-python pytesseract

# Open notebooks
jupyter notebook
```

> **Note:** The Berber-MNIST dataset is not included in this repo. You'll need to source it separately and update the file paths in the notebooks.

## Academic Context

This project was completed as part of an **MSc in Data Science** dissertation. The full project proposal is included in the repository.

## References

Key papers that informed the model selection:
- LeCun et al. (1998) — LeNet-5 and gradient-based learning
- He et al. (2016) — Deep Residual Learning (ResNet)
- Sabour et al. (2017) — Dynamic Routing Between Capsules
- Dosovitskiy et al. (2020) — Vision Transformers

---

*Code shared for academic portfolio purposes.*
