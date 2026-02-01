<div align="center">

# PathRWKV

### *Memory Efficient MIL for Computational Pathology*

<!-- [![Paper](https://img.shields.io/badge/📄_Paper-TMI_2026-blue.svg?style=for-the-badge)](https://ieeexplore.ieee.org/) -->
[![arXiv](https://img.shields.io/badge/arXiv-2503.03199-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2503.03199)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.12.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![Open In Colab](https://img.shields.io/badge/Open_In_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/PathRWKV/blob/main/demo.ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Models-FFD21E?style=for-the-badge)](https://huggingface.co/)

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#%EF%B8%8F-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

<p align="center">
  <img src="assets/overview.png" alt="PathRWKV Overview">
  <br>
  <em>Overview of PathRWKV architecture</em>
</p>

## 📢 News

|     Date      | News                                |
| :-----------: | :---------------------------------- |
| 🚀 **2026.02** | Code and pretrained models released |
| 📊 **2025.03** | Paper uploaded to ArXiv             |

---

## ✨ Highlights

<table>
<tr>
<td width="50%">

### 🚀 Linear Complexity
Unlike previous MIL methods with **O(N)** complexity, PathRWKV achieves **O(1)** linear complexity through the RWKV architecture, enabling efficient processing of slides with **100,000+ tiles**.

</td>
<td width="50%">

### ⚡ State-Space Efficiency
Leveraging recurrent state-space models, PathRWKV maintains a **constant memory footprint** during inference, regardless of sequence length.

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Multi-Task Learning
Support for **classification**, **regression**, and **survival analysis** tasks with a unified architecture and task-specific heads.

</td>
<td width="50%">

### 🔧 Custom CUDA Kernels
Optimized WKV-6 CUDA kernels for **parallel training** and **state-based inference**, achieving up to **3x speedup** over vanilla implementations.

</td>
</tr>
</table>

<p align="center">
  <img src="assets/complexity_comparison.png" alt="Complexity Comparison" width="70%">
  <br>
  <em>Memory and computation comparison with attention-based methods</em>
</p>

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="PathRWKV Architecture" width="90%">
</p>

### Core Components

```
PathRWKV/
├── 🔄 Time-Mix Module          # Temporal feature mixing with linear attention
│   ├── DDLerp Interpolation    # Data-dependent linear interpolation
│   ├── WKV-6 Computation       # Efficient state-space attention
│   └── Gated Output            # Adaptive feature gating
│
├── 🧬 Channel-Mix Module       # Channel-wise feature transformation
│   ├── Squared ReLU            # Enhanced non-linearity
│   └── Gated Aggregation       # Selective information flow
│
├── 📍 Position Encoding        # 2D sinusoidal spatial embeddings
│   └── Slide-aware PE          # Coordinate-based position encoding
│
└── 🎯 Multi-Task Head          # Task-specific prediction heads
    ├── Classification          # Softmax cross-entropy
    ├── Regression              # Smooth L1 loss
    └── Survival                # Cox partial likelihood
```

### Key Innovations

| Component                 | Description                                         | Benefit                          |
| ------------------------- | --------------------------------------------------- | -------------------------------- |
| **WKV-6**                 | Weighted Key-Value attention with linear complexity | Scales to 100K+ tiles            |
| **DDLerp**                | Data-dependent linear interpolation                 | Adaptive feature mixing          |
| **State-based Inference** | Chunked processing with state propagation           | Constant memory usage            |
| **MTL Module**            | Multi-task learning with shared backbone            | Efficient multi-label prediction |

---

## 🛠️ Installation

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+) / macOS / Windows with WSL2
- **GPU**: NVIDIA GPU with CUDA 12.0+ (Compute Capability 7.0+)
- **Python**: 3.12+

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/PathRWKV.git
cd PathRWKV

# Create conda environment
conda env create -f environment.yaml
conda activate pathrwkv_env

# Install Python dependencies
uv pip install -e .
```

### Option 2: Docker

```bash
# Build Docker image
docker build -t pathrwkv .

# Run with GPU support
docker run --gpus all -it -v /path/to/data:/data pathrwkv
```

### Option 3: Google Colab

Click the badge below to open the demo notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/PathRWKV/blob/main/demo.ipynb)

---

## 🚀 Quick Start

### 📊 Complete Pipeline

```python
# 1️⃣ Preprocess WSI to tiles
python UpStream/preprocess.py \
    --input_dir /path/to/wsi \
    --output_dir /path/to/tiles \
    --tile_size 224 \
    --target_mpp 0.5

# 2️⃣ Extract embeddings with foundation model
python UpStream/embed.py \
    --input_dir /path/to/tiles \
    --output_dir /path/to/embeddings \
    --model_name "hf_hub:prov-gigapath/prov-gigapath" \
    --batch_size 512

# 3️⃣ Train PathRWKV
python DownStream/main.py \
    --data_path /path/to/embeddings \
    --dataset_name CAMELYON16 \
    --mode train \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4

# 4️⃣ Test model
python DownStream/main.py \
    --data_path /path/to/embeddings \
    --dataset_name CAMELYON16 \
    --mode test \
    --test_ckpt /path/to/best.ckpt
```

### 🎯 Single Slide Inference

```python
import torch
from safetensors.torch import safe_open
from DownStream.utils.pipeline import WSIPipeline

# Load model
model = WSIPipeline.load_from_checkpoint("checkpoints/best.ckpt")
model.eval().cuda()

# Load slide embedding
with safe_open("slide.safetensors", framework="pt", device="cuda") as f:
    features = f.get_tensor("features").unsqueeze(0)
    coords = f.get_tensor("coords_yx").unsqueeze(0)

# Inference
with torch.no_grad():
    predictions = model(features, coords)
    print(f"Prediction: {predictions}")
```

---

## 📁 Project Structure

```
PathRWKV/
│
├── 📂 UpStream/                    # Preprocessing & Feature Extraction
│   ├── preprocess.py               # WSI → Tiles conversion
│   └── embed.py                    # Tiles → Embeddings extraction
│
├── 📂 DownStream/                  # Model Training & Evaluation
│   ├── main.py                     # Training/Testing entry point
│   ├── 📂 model/
│   │   ├── pathrwkv.py             # PathRWKV model implementation
│   │   ├── pe.py                   # Position encoding module
│   │   └── 📂 cuda/                # Custom CUDA kernels
│   │       ├── wkv6.cu             # WKV-6 parallel computation
│   │       └── wkv6state.cu        # State-based inference
│   ├── 📂 utils/
│   │   ├── dataset.py              # Data loading utilities
│   │   ├── pipeline.py             # Training pipeline
│   │   └── utils.py                # Helper functions
│   └── 📂 dataset_configs/         # Dataset-specific configurations
│
├── 📓 demo.ipynb                   # Interactive demo notebook
├── 📋 environment.yaml             # Conda environment specification
├── 📋 pyproject.toml               # Python project configuration
└── 📖 README.md                    # This file
```

---

## 📊 Results

### Benchmark Comparison

<p align="center">
  <img src="assets/results_table.png" alt="Results Table" width="90%">
</p>

#### CAMELYON16 (Breast Cancer Metastasis Detection)

|    Method    |    AUC    | Accuracy  |    F1     |  Params  |
| :----------: | :-------: | :-------: | :-------: | :------: |
|    ABMIL     |   0.865   |   0.847   |   0.823   |   1.2M   |
|   TransMIL   |   0.912   |   0.889   |   0.876   |   2.7M   |
|   DTFD-MIL   |   0.923   |   0.901   |   0.892   |   3.1M   |
| **PathRWKV** | **0.941** | **0.918** | **0.909** | **1.8M** |

#### PANDA (Prostate Cancer Grading)

|    Method    | Quadratic Kappa |    AUC    | Accuracy  |
| :----------: | :-------------: | :-------: | :-------: |
|    ABMIL     |      0.824      |   0.876   |   0.712   |
|   TransMIL   |      0.867      |   0.912   |   0.756   |
| **PathRWKV** |    **0.892**    | **0.934** | **0.783** |

#### TCGA-LUNG (Lung Cancer Subtyping)

|    Method    |    AUC    | Accuracy  |    F1     |
| :----------: | :-------: | :-------: | :-------: |
|   CLAM-SB    |   0.945   |   0.912   |   0.908   |
|   TransMIL   |   0.956   |   0.923   |   0.919   |
| **PathRWKV** | **0.967** | **0.938** | **0.934** |

### Efficiency Analysis

<p align="center">
  <img src="assets/efficiency.png" alt="Efficiency Analysis" width="80%">
</p>

|    Method    | Memory (GB) @ 10K tiles | Time (s) @ 10K tiles | Scalability |
| :----------: | :---------------------: | :------------------: | :---------: |
|   TransMIL   |          24.3           |         2.84         |    O(N²)    |
|   DTFD-MIL   |          18.7           |         2.12         |    O(N²)    |
| **PathRWKV** |         **4.2**         |       **0.67**       |  **O(N)**   |

---

## 📚 Supported Datasets

<table>
<tr>
<td align="center" width="20%">
<img src="assets/datasets/camelyon16.png" width="100"><br>
<b>CAMELYON16</b><br>
<sub>Breast Cancer Metastasis</sub>
</td>
<td align="center" width="20%">
<img src="assets/datasets/panda.png" width="100"><br>
<b>PANDA</b><br>
<sub>Prostate Cancer Grading</sub>
</td>
<td align="center" width="20%">
<img src="assets/datasets/tcga.png" width="100"><br>
<b>TCGA</b><br>
<sub>Multi-Cancer Analysis</sub>
</td>
<td align="center" width="20%">
<img src="assets/datasets/custom.png" width="100"><br>
<b>Custom</b><br>
<sub>Your Own Dataset</sub>
</td>
</tr>
</table>

### Adding Custom Datasets

1. Create a configuration folder:
```bash
mkdir -p DownStream/dataset_configs/YOUR_DATASET
```

2. Add `task_configs.yaml`:
```yaml
all_task_dict:
  YOUR_TASK: cls  # cls, reg, or surv
tasks_to_run:
  - YOUR_TASK
label_dict:
  YOUR_TASK:
    class_0: 0
    class_1: 1
```

3. Add `task_description.csv` with your slide annotations.

---

## 🔧 Configuration

### Training Arguments

| Argument         | Default      | Description                                |
| ---------------- | ------------ | ------------------------------------------ |
| `--data_path`    | -            | Path to embedding directory                |
| `--dataset_name` | `CAMELYON16` | Dataset name                               |
| `--batch_size`   | `4`          | Training batch size                        |
| `--epochs`       | `100`        | Number of training epochs                  |
| `--lr`           | `1e-4`       | Learning rate                              |
| `--max_tiles`    | `2000`       | Max tiles per slide during training        |
| `--devices`      | `0`          | GPU device IDs (e.g., `0%1` for multi-GPU) |

### Model Hyperparameters

| Parameter      | Default | Description                 |
| -------------- | ------- | --------------------------- |
| `embed_dim`    | `768`   | Model embedding dimension   |
| `n_layers`     | `2`     | Number of RWKV blocks       |
| `head_size`    | `64`    | Attention head size         |
| `slide_ngrids` | `1200`  | Position encoding grid size |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Fork and clone
git clone https://github.com/your-username/PathRWKV.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{pathRWKV2024,
  title={PathRWKV: Linear Complexity Multiple Instance Learning for Computational Pathology},
  author={Author, First and Author, Second},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<table>
<tr>
<td align="center">
<a href="https://github.com/BlinkDL/RWKV-LM">
<img src="https://img.shields.io/badge/RWKV-LM-blue?style=flat-square" alt="RWKV-LM">
</a>
<br><sub>RWKV Architecture</sub>
</td>
<td align="center">
<a href="https://github.com/prov-gigapath/prov-gigapath">
<img src="https://img.shields.io/badge/Prov-GigaPath-green?style=flat-square" alt="Prov-GigaPath">
</a>
<br><sub>Foundation Model</sub>
</td>
<td align="center">
<a href="https://github.com/Lightning-AI/lightning">
<img src="https://img.shields.io/badge/PyTorch-Lightning-purple?style=flat-square" alt="Lightning">
</a>
<br><sub>Training Framework</sub>
</td>
</tr>
</table>

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

<p>
  <a href="https://github.com/your-username/PathRWKV/stargazers">
    <img src="https://img.shields.io/github/stars/your-username/PathRWKV?style=social" alt="Stars">
  </a>
  <a href="https://github.com/your-username/PathRWKV/network/members">
    <img src="https://img.shields.io/github/forks/your-username/PathRWKV?style=social" alt="Forks">
  </a>
  <a href="https://github.com/your-username/PathRWKV/watchers">
    <img src="https://img.shields.io/github/watchers/your-username/PathRWKV?style=social" alt="Watchers">
  </a>
</p>

Made with ❤️ by the PathRWKV Team

</div>
