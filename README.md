## Overview

This repository contains a comprehensive Network Intrusion Detection System (NIDS) built using Multi-Layer Perceptron (MLP) neural networks. The system is designed to detect and classify various types of network attacks, including DDoS, Memcrashed, and Port Scanning attacks.

The project implements a complete pipeline from raw network data processing through feature extraction, feature selection, model training, and testing. It provides an interactive interface for users to easily navigate through the different stages of the intrusion detection process.

## Key Features

- **Complete Pipeline**: End-to-end solution from raw packet data to trained models
- **Multi-attack Detection**: Handles DDoS, Memcrashed, and Port Scanning attacks
- **Feature Engineering**: Extracts 16 network traffic features optimized for attack detection
- **Feature Selection**: Implements correlation analysis and Random Forest-based feature importance
- **Deep Learning Model**: Uses MLP with attention mechanism for improved classification
- **Interactive Interface**: User-friendly command-line interface to access all functions
- **Comprehensive Documentation**: Detailed documentation for each component
- **Performance Visualization**: Generates detailed performance metrics and visualizations

## System Architecture

The system consists of five main components:

1. **Feature Extraction**: Processes raw PCAP files to extract network traffic features
2. **Feature Labelling**: Labels network traffic data with attack types
3. **Feature Selection**: Identifies the most relevant features for attack detection
4. **Model Training**: Trains an MLP neural network with the selected features
5. **Model Testing**: Evaluates the trained model on new network traffic data

## Installation Requirements

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TKinter (for GUI file selection)
- Scapy (for packet processing)
- ipaddress
- scipy

Install dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scapy ipaddress scipy
```

## Usage

The main interface is provided through `MLP_model.py`, which gives access to all components of the system:

```bash
python MLP_model.py
```

This will launch an interactive menu with the following options:
1. Feature Extraction
2. Feature Selection
3. Train the Model
4. Test the Model
5. Exit

## Detailed Documentation

Each component has comprehensive documentation:

- [Feature Extraction](../../wiki/feature_extraction) - Extracting network traffic features from PCAP files
- [Feature Labelling](../../wiki/feature_labelling) - Labelling network traffic with attack types
- [Feature Selection](../../wiki/feature_selection) - Selecting the most relevant features
- [Model Training](../../wiki/model_train) - Training the MLP neural network model
- [Model Testing](../../wiki/model_test) - Testing and evaluating the trained model

## Component Overview

### Feature Extraction

The feature extraction module processes raw network packet data from PCAP files to extract relevant features for intrusion detection. It includes:

- Efficient batch processing for large PCAP files
- Parallel processing using multi-threading
- Comprehensive feature extraction including flow-based, connection-based, flag-based, and window-based features
- Memory optimization techniques
- Progress tracking with real-time updates

[Read more about Feature Extraction](../../wiki/feature_extraction)

### Feature Labelling

This module handles the labeling of network traffic data with different attack types:

- Attack labeling based on XML definition files
- Support for multiple attack types (DDoS, Memcrashed, Port Scan)
- Conversion of categorical features to numerical values
- Data normalization for optimal model performance
- Comprehensive statistics on attack distribution

[Read more about Feature Labelling](../../wiki/feature_labelling)

### Feature Selection

The feature selection module identifies the most relevant features for intrusion detection:

- Correlation analysis to identify redundant features
- Target correlation assessment for feature relevance
- Random Forest-based feature importance evaluation
- Multi-stage filtering for optimal feature subset
- Visualization tools for feature analysis

[Read more about Feature Selection](../../wiki/feature_selection)

### Model Training

The model training module implements a neural network approach for intrusion detection:

- Custom IDSNet neural network architecture with attention mechanism
- Class imbalance handling with multiple balancing strategies
- Data augmentation with feature and label noise
- Comprehensive performance metrics and visualization
- Model calibration for optimal classification thresholds

[Read more about Model Training](../../wiki/model_train)

### Model Testing

The model testing module evaluates the trained model on new network traffic data:

- Batch processing for memory-efficient testing
- Comprehensive evaluation with detailed metrics
- Rich visualizations of model performance
- Feature analysis for model interpretation
- Support for both labeled and unlabeled data

[Read more about Model Testing](../../wiki/model_test)

## Attack Types

The system is designed to detect and classify the following attack types:

1. **Normal** (Class 0): Benign network traffic
2. **DDoS** (Class 1): Distributed Denial of Service attacks
3. **Memcrashed Spoofer** (Class 2): Amplification attacks using memcached servers
4. **PortScan** (Class 3): Network reconnaissance through port scanning

## Directory Structure

```
IDS-Project/
├── Code/
│   ├── MLP_model.py             # Main interface
│   ├── feature_extraction.py    # Feature extraction module
│   ├── feature_labelling.py     # Feature labelling module
│   ├── feature_selection.py     # Feature selection module
│   ├── model_train.py           # Model training module
│   └── model_test.py            # Model testing module
├── trained_model/               # Directory for saved models
├── Results/                     # Directory for results and visualizations
├── Logs/                        # Directory for log files
├── wiki/                        # Wiki documentation 
│   ├── Home.md
│   ├── feature_extraction.md
│   ├── feature_labelling.md
│   ├── feature_selection.md
│   ├── model_train.md
│   └── model_test.md
```




