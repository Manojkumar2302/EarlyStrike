# EarlyStrike – Hybrid AI-Based Ransomware Detection System

## Overview

EarlyStrike is a research-oriented prototype that demonstrates the application of Artificial Intelligence for behavioral ransomware detection. The system combines anomaly detection and sequence learning techniques with real-time system monitoring to identify potentially malicious activity at an early stage.

The project integrates machine learning, system monitoring, feature extraction, alert management, and an interactive dashboard into a unified framework. It was developed as an undergraduate capstone and research project to explore AI-driven cybersecurity solutions.

> **Note:** EarlyStrike is intended as a proof-of-concept and research prototype. It is not designed to replace commercial endpoint security or antivirus solutions.

---

## Key Features

### Machine Learning

* Autoencoder-based anomaly detection
* CNN-BiLSTM behavioral classification
* Hybrid threat assessment
* Model training and evaluation pipeline
* Prediction pipeline for new samples

### Real-Time Monitoring

* Windows system monitoring
* Behavioral feature extraction
* Process and resource monitoring
* Suspicious activity detection
* Continuous system analysis

### Dashboard

* Real-time monitoring interface
* System health visualization
* Threat score monitoring
* Detection logs
* Alert history
* Live system statistics

### Alert Management

* SQLite-based alert storage
* Threat prioritization
* Email notification support
* Webhook integration
* Alert acknowledgment and history

---

## System Architecture

```
Windows System
      │
      ▼
Windows Feature Extractor
      │
      ▼
System Monitoring Agent
      │
      ▼
Feature Engineering
      │
 ┌────┴────┐
 ▼         ▼
Autoencoder    CNN-BiLSTM
      │
      ▼
Hybrid Detection Engine
      │
      ▼
Alert Management
      │
      ▼
Flask Dashboard
```

---

## Project Structure

```
EarlyStrike/

├── dashboard/
│   ├── index.html
│   ├── script.js
│   ├── simple_dashboard_backend.py
│   └── launch_improved_dashboard.py
│
├── monitoring/
│   ├── system_monitor_agent.py
│   ├── windows_feature_extractor.py
│   └── alert_system.py
│
├── models/
│   ├── autoencoder_model.py
│   ├── cnn_bilstm_model.py
│   └── ransomware_detection.py
│
├── training/
│   ├── train.py
│   ├── train_autoencoder.py
│   ├── evaluate_realistic_models.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

**Programming Languages**

* Python
* JavaScript
* HTML
* CSS

**Machine Learning**

* TensorFlow
* Scikit-Learn
* NumPy
* Pandas

**Backend**

* Flask
* Flask-CORS

**Monitoring**

* psutil
* watchdog

**Database**

* SQLite

**Visualization**

* Matplotlib
* Seaborn
* Chart.js

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Manojkumar2302/EarlyStrike.git
cd EarlyStrike
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python launch_improved_dashboard.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

## Detection Pipeline

EarlyStrike combines two complementary machine learning models:

### Autoencoder

* Learns normal system behavior
* Detects anomalous activities based on reconstruction error

### CNN-BiLSTM

* Learns behavioral sequences from system events
* Identifies ransomware activity through sequence classification

The outputs from both models are combined to estimate the final threat level before generating alerts.

---

## Current Status

EarlyStrike is a functional research prototype developed to explore behavioral ransomware detection using machine learning.

The project demonstrates the integration of:

* Behavioral feature extraction
* Real-time monitoring
* Hybrid machine learning models
* Threat assessment
* Alert management
* Interactive visualization

Although fully functional as a prototype, it has not been evaluated for production deployment or enterprise-scale environments.

---

## Current Limitations

* Currently focused on Windows-based behavioral monitoring.
* Detection performance depends on the quality and diversity of the training dataset.
* The system has not undergone extensive enterprise-scale validation.
* Some behavioral heuristics can be further refined.
* Production-level security hardening, scalability, and automated testing remain future work.

---

## Future Improvements

Potential future enhancements include:

* Windows ETW integration
* Explainable AI (XAI)
* Incremental model learning
* Cross-platform support
* Cloud-based monitoring
* Docker deployment
* REST API authentication
* CI/CD integration
* Expanded ransomware datasets
* Performance optimization for large-scale environments

---

## Disclaimer

This repository is provided for **educational, academic, and research purposes**.

EarlyStrike is a research prototype developed to demonstrate AI-based behavioral ransomware detection and should not be considered a replacement for commercial endpoint protection or antivirus software.

The project is intended to serve as a foundation for further research and future improvements.

---

## Author

**Rajoli Manoj Kumar Reddy**

Bachelor of Technology (Information Technology)

Research Interests:

* Artificial Intelligence
* Cybersecurity
* Machine Learning
* Malware & Ransomware Detection
