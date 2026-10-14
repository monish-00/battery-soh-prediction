# AI-Based Battery State-of-Health Prediction

## 1. Overview

This project aims to develop a machine learning-based system
for estimating the State of Health (SoH) of lithium-ion batteries.

The system will use battery degradation datasets to develop
and evaluate ML models. It will subsequently integrate software
battery simulation and hardware simulation to demonstrate how
battery measurements can be processed for SoH estimation.

## 2. Objectives

- Study battery degradation and State of Health estimation.
- Explore publicly available battery aging datasets.
- Develop and evaluate machine learning models for SoH prediction.
- Simulate battery charging, discharging, and degradation.
- Simulate sensor-based data acquisition using a microcontroller.
- Integrate the prediction model with a visualization dashboard.
- Develop a low-cost, low-power physical prototype as the final phase.

## 3. Planned Data Sources

Primary dataset:
NASA Li-ion Battery Aging Dataset

Potential secondary dataset:
CALCE Battery Research Data

## 4. Development Phases

1. Literature survey and project definition
2. Dataset exploration and preprocessing
3. ML model development and evaluation
4. Software battery simulation
5. Hardware simulation
6. Backend and dashboard integration
7. System testing and validation
8. Physical hardware implementation

## 5. Planned Technology Stack

Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook

Additional technologies for simulation, backend, dashboard,
and hardware will be selected during development.

## 6. Important Research Considerations

- Evaluate models using appropriate regression metrics.
- Prevent data leakage during training and testing.
- Distinguish real experimental data from simulated data.
- Document assumptions and limitations.
- Validate the prediction pipeline before physical hardware
  implementation.