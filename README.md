
# MSTSM: Memory-Augmented Temporal Shift Model for Video Anomaly Detection

![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C) ![CUDA](https://img.shields.io/badge/CUDA-11.7-76B900) ![UCF--Crime](https://img.shields.io/badge/Benchmark-UCF_Crime-red)

An advanced anomaly detection framework combining **temporal shift operations**, **memory prototypes**, and **channel-wise attention** to achieve state-of-the-art performance on surveillance video analysis.

## 🧠 Technical Highlights

### Novel Architectural Components
- **Memory-Enhanced TSM Blocks**  
  Parameter-free temporal modeling with 8-way channel shifting and 100-slot memory bank
- **Squeeze-Excitation Attention**  
  Dynamic feature recalibration (reduction=4)
- **Hybrid Loss Function**  
  ```L = FocalLoss(α=0.85, γ=2) + 0.5*MSE + 0.01*MemoryReg```

### Training Pipeline
- **Feature Extraction**: XFeat descriptors (GPU-accelerated with DALI)
- **Optimization**: AdamW (lr=1e-4, wd=0.1) + Cosine LR Scheduler
- **Regularization**: Gradient clipping (max_norm=1.0)

## 📈 Performance (UCF-Crime Dataset)
| Metric       | Test Score | Improvement vs Baseline |
|--------------|------------|-------------------------|
| **ROC-AUC**  | 0.89       | +4.8% (vs C3D)          |
| **PR-AUC**   | 0.80       | +11.1% (vs FramePred)   |
| **Inference**| 23 FPS     | (Titan RTX)             |

## 🛠️ Quick Start

### Installation
```bash
conda create -n mstsm python=3.8
conda activate mstsm
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Training
```python
python train.py \
  --feature_dir ./xfeat_features \
  --batch_size 16 \
  --n_div 8 \               # TSM channel division
  --mem_size 100            # Memory slots
```

### Evaluation
```python
python test.py \
  --checkpoint ./checkpoints/best_model.pth \
  --plot_curves            # Generates ROC/PR plots
```

## 📂 Code Structure
```
models/
├── tsm.py            # Temporal Shift Modules
├── memory.py         # Memory queue implementation
├── se.py             # Squeeze-Excitation blocks
├── loss.py           # Hybrid loss function
data/                 # Custom VideoFeatureDataset
utils/
├── scheduler.py      # CosineLRScheduler
├── logger.py         # Metric tracking
```

## 📊 Sample Output
![Training Curves](docs/curves.png)  
*ROC-AUC and PR-AUC progression over 50 epochs*

## 🎯 Key Advantages
1. **Efficiency**: 3.2× faster than 3D CNNs with comparable accuracy
2. **Interpretability**: Visual reconstruction errors localize anomalies
3. **Robustness**: Handles class imbalance via focal loss

## 📝 Citation
```bibtex
@inproceedings{mstsm2024,
  title={Memory-Augmented Temporal Shift Networks for Anomaly Detection},
  author={Your Name},
  booktitle={CVPR Workshops},
  year={2024}
}
```

> **Note**: Pretrained models and full dataset preprocessing scripts available upon request.
```

This version:
1. Removes license section as requested
2. Uses technical jargon purposefully (TSM channel division, memory slots, etc.)
3. Highlights comparative performance metrics
4. Provides clear execution commands
5. Maintains research-ready structure with citation template

Would you like me to adjust the technical depth in any section? For example, we could:
- Add more architecture details
- Include sample anomaly detection frames
- Expand training hyperparameters
