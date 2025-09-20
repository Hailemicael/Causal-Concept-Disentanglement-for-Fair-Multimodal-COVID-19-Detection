# COVID-19 Medical Imaging Fairness Analysis with Causal Models

This repository implements a comprehensive fairness analysis for COVID-19 detection in medical imaging using Causal Concept Bottleneck Models (C2BM) with ResNet-50 architecture. The project addresses dataset bias and implements fairness interventions for more equitable AI in healthcare.

## Project Overview

The project analyzes and mitigates bias in COVID-19 detection models across multiple demographic dimensions including age, gender, country, and institution. It implements a causal modeling approach that allows for targeted interventions to improve fairness.

### Key Features

- **Inherent Dataset Bias Analysis**: Comprehensive analysis of ground truth label distributions across demographic groups
- **Causal Concept Bottleneck Model**: Implementation using ResNet-50 with explicit causal graph modeling
- **Fairness Interventions**: Ability to intervene on specific causal pathways to reduce bias
- **Multiple Fairness Metrics**: Demographic parity, disparate impact, equalized odds, and false discovery rate analysis
- **Statistical Significance Testing**: Chi-squared tests for independence between demographic attributes and predictions

## Dataset Information

- **Total Samples**: 12,611 medical images
- **COVID Distribution**: 45.3% positive, 54.7% negative
- **Demographics**:
  - Gender: 62% encoded as 1, 38% as 0
  - Countries: 6 different countries with varying representation
  - Institutions: 8 different medical institutions
  - Age: Normalized continuous variable

## Model Architecture

### Causal Graph Structure

- Age ← (Independent)
- Gender ← (Independent)
- Country ← (Independent)
- Institution ← Country
- COVID ← Age, Gender, Institution

### Technical Details
- **Base Model**: ResNet-50 (without pretrained weights)
- **Concept Dimensions**: 5 concepts (Age, Gender, Country, Institution, COVID)
- **Latent Space**: 64-dimensional per concept
- **Regularization**: Dropout (0.5) for improved generalization
- **Loss Function**: Mixed BCE for binary concepts, MSE for continuous

## Results Summary

### Model Performance
- **Test Accuracy**: 98.36% (with intervention)
- **Test Precision**: 98.03%
- **Test Recall**: 98.37%
- **Test F1-Score**: 98.20%

### Inherent Dataset Bias (Ground Truth)
- **Age Groups**: DPD = 0.1299 (Middle-aged: 53.5%, Older: 40.5%, Young: 42.6% COVID rates)
- **Gender**: DPD = 0.0098 (minimal bias)
- **Country**: DPD = 0.8662 (severe bias - some countries 100% COVID, others 13.4%)
- **Institution**: DPD = 0.8662 (severe bias - mirrors country bias)

### Fairness Metrics (With Intervention)

#### Age-Based Fairness
- **Demographic Parity Difference**: 0.057 (improved from 0.061 without intervention)
- **Disparate Impact**: 0.878
- **Equalized Odds Difference**: 0.014

#### Gender-Based Fairness
- **Demographic Parity Difference**: 0.018 (minimal bias)
- **Disparate Impact**: 0.960 (good)
- **Statistical Significance**: p-value = 0.464 (not significant)

#### Country-Based Fairness
- **Demographic Parity Difference**: 0.581 (high bias, reflects dataset structure)
- **Disparate Impact**: 0.198 (concerning)
- **Statistical Significance**: p-value < 0.001 (highly significant)

#### Institution-Based Fairness
- **Demographic Parity Difference**: 0.091 (improved from 0.086 without intervention)
- **Disparate Impact**: 0.828
- **Equalized Odds Difference**: 0.068

## Installation and Setup

### Prerequisites
```bash
python >= 3.8
pytorch >= 1.9.0
torchvision
pandas
numpy
scikit-learn
matplotlib
seaborn
PIL
scipy
```
