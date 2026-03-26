# maml-korean-winter-forecast
MAML-based few-shot learning framework for seasonal prediction of Korean winter temperature using domain-knowledge-driven data augmentation.

This repository provides the implementation of a MAML-based convolutional neural network for seasonal prediction of Korean winter temperature anomalies.

The framework addresses the limited sample problem in climate forecasting by combining:

model-agnostic meta-learning (MAML)
domain-knowledge-based data augmentation using climate indices (e.g., AO, SH, Niño3.4)

The proposed approach demonstrates improved forecast skill compared to conventional CNN-based models and dynamical forecast systems, particularly for 1–3 month lead predictions during boreal winter.
