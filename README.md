# DMIL: Decomposition-based Multimodal Interaction Learning

Official implementation of Information-Theoretic Decomposition for Multimodal Interaction Learning (DMIL) (CVPR 2026).

> Paper: [CVPR 2026 Open Access (PDF)](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.pdf) | [arXiv:2606.11614 (PDF)](https://arxiv.org/pdf/2606.11614)

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

The default schedule follows the paper: 10 epochs of joint intra-modality
decomposition, 5 epochs of consistency decomposition with the Stage-1 modules
frozen, and joint fine-tuning for the remaining epochs. Set
`methods.stage2_epochs=0` to run Stage 1 only.

## Citation

```bibtex
@inproceedings{yang2026information,
  title     = {Information-Theoretic Decomposition for Multimodal Interaction Learning},
  author    = {Yang, Zequn and Wei, Yake and Ni, Haotian and Xu, Zhihao and Hu, Di},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
