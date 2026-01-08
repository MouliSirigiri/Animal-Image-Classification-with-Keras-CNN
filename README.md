# Animal-Image-Classification-with-Keras-CNN

#**Overview**

This repository contains the code and analysis for an Applied Data Science assignment on building a Convolutional Neural Network (CNN) using Keras/TensorFlow to classify images of animals into three categories: cats, dogs, and wild animals. The project processes a custom dataset from Google Drive (train/val splits), defines a multi-layer CNN architecture, trains it with callbacks for optimization, and evaluates performance via loss/accuracy plots and per-class metrics. Leveraging GPU acceleration in Colab, it achieves ~85% validation accuracy after 100 epochs, demonstrating effective feature extraction for image tasks.

#**Key goals:**

Efficiently load and preprocess images (resize to 64x64, float32 normalization).
Architect a CNN with Conv2D layers, GlobalAvgPool2D, and Dense classifiers.
Train with Adam (lr=5e-5), sparse categorical crossentropy, and callbacks (checkpoint, LR scheduler).
Evaluate via training curves and confusion matrices; discuss overfitting and class balance.

Insights: Model excels on cats/dogs (F1>0.88) but struggles with wild (F1=0.75) due to intra-class variance; early stopping via checkpoint prevents degradation.

#**Key Findings**

Training Curves: Loss plateaus ~epoch 40; val acc peaks at 86% (checkpoint saves here). Slight overfitting (train-val gap ~5%) mitigated by LR decay.

**Per-Class Performance:**

Class,Precision,Recall,F1-Score
Cat,0.89,0.92,0.90
Dog,0.87,0.88,0.88
Wild,0.78,0.72,0.75

Insights: Conv layers capture edges/textures well for pets; wild variability (e.g., lions vs. birds) confuses model. Augmentation (flips/rotations) could boost wild F1 +10%.
Critical Eval: Sparse loss suits integer labels; GlobalAvgPool reduces params (363k total). Future: Transfer learning (ResNet) for 90%+ acc; report discusses bias toward common breeds.

#**Data Sources**

Primary: Custom Animals dataset (Google Drive; ~1.2k JPEGs, 224x224 orig.).
Classes: Cat (pets), Dog (breeds), Wild (lions, eagles, etc.).
Splits: 80/20 train/val; balanced.


#**Methods**

Preprocessing: tf.data.Dataset with map (resize/normalize), cache/shuffle/prefetch for speed.
Model: 6 Conv2D (ReLU, strides=2 halving dims), GlobalAvgPool, Dense (512→3 Softmax).
Training: 100 epochs; Adam, sparse_xentropy; callbacks for best weights/LR decay.
Eval: History plots, predictions on val; sklearn for metrics/confusion.

#**Limitations and Future Work**

Small dataset risks overfitting; extend to CIFAR-10 subset.
No aug/test set; add for robustness.
Report: Analyzes curves (underfit early, overfit late); suggests dropout/batch norm.
