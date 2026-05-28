# Multimodal-RTLI-Prediction

This repository accompanies the manuscript:

> Multimodal Deep Learning-Based Prediction of Radiation-Induced Temporal Lobe Injury via Anatomical Imaging and Spatial Dose: A Multicenter Study

## Overview

This repository provides an implementation of the proposed multimodal deep learning framework for radiation-induced temporal lobe injury (RTLI) prediction using:

- anatomical imaging
- radiation dose distribution
- ROI mask information
- structured feature integration

The repository is provided for:

- academic transparency
- methodological reference
- research communication

------

## Repository Structure

```text
dataset.py
model.py
resnet3d.py
inference.py
requirements.txt
README.md
```

------

## Expected Input Format

### CSV

|id|feature_1|feature_2|...|

The CSV file contains structured clinical and dosimetric features used for model inference.

### NPZ

Each case should contain:

```text
ct : anatomical image volume
rd : radiation dose volume
rs : ROI mask volume
```

------

## Model

The framework includes:

- 3D ResNet-based encoder
- multimodal feature fusion
- structured feature integration
- survival prediction head

The output dimension corresponds to discretized survival intervals used in the prediction framework.

------

## Example

```bash
python inference.py
```

------

## Notes

Patient-level data are not publicly available due to institutional data governance restrictions.

This repository provides an inference-oriented implementation for academic transparency and methodological reference.