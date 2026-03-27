# DMIL: Decomposition-based Multimodal Interaction Learning

Official implementation of Information-Theoretic Decomposition for Multimodal Interaction Learning (DMIL) (CVPR 2026).

> Paper: [CVPR 2026 — link TBD] | arXiv: [link TBD]

## Method

Multimodal data contains three types of interaction between modalities: **Redundancy** (information shared by both), **Uniqueness** (information exclusive to each modality), and **Synergy** (information that only emerges from their joint consideration). DMIL explicitly decomposes multimodal representations into these components via a hierarchical variational bottleneck and learns them through a dynamic gating mechanism, enabling the model to adapt to the specific interaction composition of each sample.

## Installation

```bash
pip install torch torchvision torchaudio librosa hydra-core scikit-learn pandas openpyxl
```

## Data Preparation

Download [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and organize as:

```
/path/to/CREMAD/
├── train.csv
├── test.csv
├── AudioWAV/
│   └── <file_id>.wav
└── Image-05-FPS/
    └── <file_id>/
        └── *.jpg   (frames extracted at 5 FPS)
```

Then set your paths in `cfgs/data_paths.yaml`:

```yaml
cremad:
  data_root: /path/to/CREMAD
  visual_feature_path: /path/to/CREMAD/Image-05-FPS
  audio_feature_path: /path/to/CREMAD/AudioWAV
```

## Quick Start

**Training:**
```bash
python main.py dataset=CREMAD methods=DMIL
```

## Citation

```bibtex
@inproceedings{yang2026information,
  title     = {Information-Theoretic Decomposition for Multimodal Interaction Learning},
  author    = {Yang, Zequn and Wei, Yake and Ni, Haotian and Xu, Zhihao and Hu, Di},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
