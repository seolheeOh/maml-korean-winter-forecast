# maml-korean-winter-forecast

Official implementation of the paper:
"Few shot learning for Korean winter temperature forecasts" 
(https://www.nature.com/articles/s41612-024-00813-z)

MAML-based few-shot learning framework for seasonal prediction of Korean winter temperature using domain-knowledge-driven data augmentation. This repository provides the implementation of a MAML-based convolutional neural network for seasonal prediction of Korean winter temperature anomalies.

The framework addresses the limited sample problem in climate forecasting by combining:

- model-agnostic meta-learning (MAML)
- domain-knowledge-based data augmentation using climate indices (e.g., AO, SH, Niño3.4)

The proposed approach demonstrates improved forecast skill compared to conventional CNN-based models and dynamical forecast systems, particularly for 1–3 month lead predictions during boreal winter.

## Requirements / Environment
- Python 3
- Tensorflow

The implementation follows the setup described in the paper.
See `requirements.txt` for full dependencies.

## Citation
@article{oh2024fewshot,
  title={Few-shot learning for Korean winter temperature forecasts},
  author={Oh, Seol-Hee and Ham, Yoo-Geun},
  journal={npj Climate and Atmospheric Science},
  year={2024}
}
