"""
Aditya-L1 Solar Flare Early Warning System — ML Pipeline.

This package contains data loading, preprocessing, augmentation,
model definitions, training scripts, and evaluation utilities for
three complementary anomaly-detection / forecasting models:

1. **Denoising Autoencoder** – unsupervised reconstruction-error anomaly detector.
2. **CNN Nowcaster** – binary flare-in-progress classifier on HEL1OS hard X-ray.
3. **BiLSTM Forecaster** – multi-task flare-class + time-to-peak predictor on SoLEXS.
"""
