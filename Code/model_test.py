import os
import logging
import traceback
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# Filter specific sklearn warnings that clutter the console output
# This will redirect these specific warnings to the log file instead
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")
warnings.filterwarnings("ignore", message="No positive samples in y_true, true positive value should be meaningless")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model") 
LOG_DIR = os.path.join(PROJECT_ROOT, "Logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")
FEATURE_FILE = os.path.join(PROJECT_ROOT, "Code", "feature_dict.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Create directories if they don't exist
for directory in [LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Attack type mapping
ATTACK_TYPES = {
    0: 'Normal',
    1: 'DDoS',
    2: 'Memcrashed',
    3: 'PortScan'
}

# Setup device for the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Network architecture - must match the architecture used in training
class IDSNet(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(IDSNet, self).__init__()
        
        # Attention layer
        self.attention = nn.Linear(input_size, input_size)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc_out = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        # Attention mechanism
        attention_weights = self.attention_softmax(self.attention(x))
        x = x * attention_weights
        
        # Pass through the rest of the network
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

def configure_logging(csv_file):
    """Set up logging configuration with input filename"""
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    log_file = os.path.join(LOG_DIR, f"test_{base_filename}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file

def load_model():
    """Load the trained model"""
    try:
        # Load feature list to get input size
        with open(FEATURE_FILE, 'r') as f:
            feature_list = json.load(f)
        
        # Create model with correct input size
        model = IDSNet(len(feature_list))
        
        # Load best model weights
        model_path = os.path.join(MODEL_DIR, "best_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logging.info(f"Model loaded from {model_path}")
        return model, feature_list
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def load_scaler():
    """Load the scaler used to preprocess training data"""
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logging.info("StandardScaler loaded successfully")
            return scaler
        except Exception as e:
            logging.error(f"Error loading scaler: {str(e)}")
            raise
    else:
        error_msg = f"Scaler file not found at {SCALER_PATH}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

def load_and_preprocess_data(csv_file, feature_list):
    """Load and preprocess dataset for prediction"""
    try:
        # Load data
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded dataset: {len(df)} rows with {len(df.columns)} columns")
        
        # Check if all required features are present
        missing_features = [feat for feat in feature_list if feat not in df.columns]
        if missing_features:
            logging.info(f"Adding missing features to dataset: {missing_features}")
            for feat in missing_features:
                df[feat] = 0
        
        # Ensure the 'attack' column is preserved if it exists
        if 'attack' in df.columns:
            required_columns = feature_list + ['attack']
        else:
            required_columns = feature_list
        
        # Filter the dataset to include only the required columns
        df = df[[col for col in required_columns if col in df.columns]]
        
        # Convert all numeric columns
        numeric_cols = [col for col in df.columns if col != 'attack']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df.fillna(0, inplace=True)
        
        # Extract features and target if available
        X = df[feature_list].values
        y = None
        if 'attack' in df.columns:
            y = df['attack'].values.astype(int)
            logging.info("Attack labels found in dataset")
            
        return X, y, df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def predict(model, X, batch_size=128):
    """Make predictions automatically based on the highest probability."""
    model = model.to(device)
    model.eval()
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Create dataloader for batched prediction
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    # Make predictions
    with torch.no_grad():
        for (inputs,) in dataloader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            # Store probabilities
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Take the class with the highest probability
            batch_preds = torch.argmax(probabilities, dim=1)
            all_predictions.extend(batch_preds.cpu().numpy())
    
    logging.info("Predictions made automatically based on the highest probability.")
    return np.array(all_predictions), np.array(all_probabilities)

def create_output_directory(csv_file):
    """Create input-file-based output directory for results with timestamp"""
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Filter out numbers, keep letters and underscores
    string_only = ''.join([c for c in base_filename if not c.isdigit()])
    
    # Clean up consecutive underscores but keep the structure
    while '__' in string_only:
        string_only = string_only.replace('__', '_')
    
    # Remove leading/trailing dashes or underscores
    string_only = string_only.strip('_').strip('-').strip()
    
    # If string_only is empty, use a default name
    if not string_only:
        string_only = "dataset"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, f"test_{string_only}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created results directory: {output_dir}")
    return output_dir

def evaluate_and_visualize(y_true, y_pred, probabilities, output_dir, df, csv_file, feature_list):
    """Evaluate model performance and create essential visualizations"""
    
    # Classification report if ground truth is available
    if y_true is not None:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        logging.info("\nClassification Report:\n" + 
                     classification_report(y_true, y_pred, zero_division=0))
        
        # Generate and save metrics table (text format only)
        generate_metrics_table(y_true, y_pred, output_dir, csv_file)
    
    # Create summary file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Test performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {os.path.basename(csv_file)}\n")
        f.write(f"Number of samples: {len(y_pred):,d}\n\n")
        
        f.write("Prediction distribution:\n")
        for pred, count in zip(*np.unique(y_pred, return_counts=True)):
            attack_name = ATTACK_TYPES.get(pred, f"Unknown ({pred})")
            percentage = (count / len(y_pred)) * 100
            f.write(f"- {attack_name}: {count:,d} samples ({percentage:.2f}%)\n")
            
        if y_true is not None:
            f.write("\nClassification summary:\n")
            for class_id in range(len(ATTACK_TYPES)):
                class_name = ATTACK_TYPES.get(class_id, f"Class {class_id}")
                true_pos = np.sum((y_true == class_id) & (y_pred == class_id))
                actual_count = np.sum(y_true == class_id)
                pred_count = np.sum(y_pred == class_id)
                recall = true_pos / max(actual_count, 1)
                precision = true_pos / max(pred_count, 1)
                f.write(f"- {class_name}:\n")
                f.write(f"  Actual: {actual_count:,d}, Predicted: {pred_count:,d}\n")
                f.write(f"  Precision: {precision:.4f}, Recall: {recall:.4f}\n")
    
    logging.info(f"Summary saved to {summary_path}")
    
    # Generate ROC curves and PR curves if we have ground truth labels
    if y_true is not None:
        roc_auc = plot_roc_curves(y_true, probabilities, output_dir)
        pr_auc = plot_pr_curves(y_true, probabilities, output_dir)
        
        # Add ROC AUC to the summary file
        with open(summary_path, 'a') as f:
            f.write("\nROC AUC Scores:\n")
            f.write(f"Micro-average: {roc_auc['micro']:.4f}\n")
            f.write(f"Macro-average: {roc_auc['macro']:.4f}\n")
            
            f.write("\nPrecision-Recall AUC Scores:\n")
            f.write(f"Micro-average: {pr_auc['micro']:.4f}\n")
            for i in range(len(ATTACK_TYPES)):
                class_name = ATTACK_TYPES.get(i, f"Class {i}")
                f.write(f"{class_name}: {pr_auc[i]:.4f}\n")
    
    return output_dir

def generate_metrics_table(y_true, y_pred, output_dir, csv_file):
    """
    Generate and save a table with precision, recall, f1-score, and support metrics
    for each class in a human-readable format.
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - output_dir: Directory to save the metrics table
    - csv_file: Path to the input CSV file
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    logging.info("Generating detailed metrics table")
    
    # Calculate metrics for each class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate support for each class
    cm = confusion_matrix(y_true, y_pred)
    support = np.sum(cm, axis=1)  # Sum across rows for true count per class
    
    # Calculate averages
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Create a nicely formatted text version
    text_path = os.path.join(output_dir, "metrics_table.txt")
    with open(text_path, 'w') as f:
        # Add input file name at the top
        f.write(f"Dataset: {os.path.basename(csv_file)}\n")
        f.write(f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Classification Metrics\n")
        f.write("=====================\n\n")
        f.write(f"{'Class':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>10}\n")
        f.write("-" * 65 + "\n")
        
        # Per-class metrics
        for i in range(len(ATTACK_TYPES)):
            class_name = ATTACK_TYPES.get(i, f"Class {i}")
            f.write(f"{class_name:<15} | {precision[i]:>10.4f} | {recall[i]:>10.4f} | {f1[i]:>10.4f} | {support[i]:>10.0f}\n")
        
        # Average metrics
        f.write("-" * 65 + "\n")
        f.write(f"{'macro avg':<15} | {macro_precision:>10.4f} | {macro_recall:>10.4f} | {macro_f1:>10.4f} | {len(y_true):>10.0f}\n")
        f.write(f"{'micro avg':<15} | {micro_precision:>10.4f} | {micro_recall:>10.4f} | {micro_f1:>10.4f} | {len(y_true):>10.0f}\n")
        f.write(f"{'weighted avg':<15} | {weighted_precision:>10.4f} | {weighted_recall:>10.4f} | {weighted_f1:>10.4f} | {len(y_true):>10.0f}\n")
    
    logging.info(f"Metrics table saved to {text_path}")

def plot_roc_curves(y_true, probabilities, output_dir):
    """
    Generate and save ROC curves for each class and micro/macro averaged curves.
    
    Parameters:
    - y_true: Ground truth labels
    - probabilities: Predicted probabilities for each class
    - output_dir: Directory to save the ROC curve plots
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    logging.info("Generating ROC curves")
    
    # Convert to binary format for multi-class ROC
    n_classes = len(ATTACK_TYPES)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), np.array(probabilities).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average and macro-average ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (area = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (area = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple']
    for i in range(n_classes):
        class_name = ATTACK_TYPES.get(i, f"Class {i}")
        plt.plot(fpr[i], tpr[i], 
                 label=f'ROC for {class_name} (area = {roc_auc[i]:.3f})',
                 color=colors[i % len(colors)], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    logging.info(f"ROC curves saved to {roc_path}")
    return roc_auc

def plot_pr_curves(y_true, probabilities, output_dir):
    """
    Generate and save precision-recall curves for each class and micro-average.
    
    Parameters:
    - y_true: Ground truth labels
    - probabilities: Predicted probabilities for each class
    - output_dir: Directory to save the PR curve plots
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    logging.info("Generating Precision-Recall curves")
    
    # Convert to binary format for multi-class PR
    n_classes = len(ATTACK_TYPES)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute PR curve and PR area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    # Handle warnings more gracefully
    for i in range(n_classes):
        # Check if this class has any positive samples
        if np.sum(y_bin[:, i]) > 0:  # If there are positive samples
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probabilities[:, i])
            pr_auc[i] = average_precision_score(y_bin[:, i], probabilities[:, i])
        else:
            # Handle the case of no positive samples more explicitly
            logging.info(f"No positive samples for class {i} ({ATTACK_TYPES.get(i, 'Unknown')}). Setting perfect PR curve.")
            precision[i] = np.array([1.0, 1.0])
            recall[i] = np.array([0.0, 1.0])
            pr_auc[i] = 1.0  # Perfect AUC
    
    # Calculate micro-average PR curve - this should be fine even with empty classes
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), np.array(probabilities).ravel())
    pr_auc["micro"] = average_precision_score(y_bin.ravel(), np.array(probabilities).ravel())
    
    # Plot PR curves
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average PR curve
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average PR (AP = {pr_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot PR curve for each class
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple']
    for i in range(n_classes):
        class_name = ATTACK_TYPES.get(i, f"Class {i}")
        plt.plot(recall[i], precision[i], 
                 label=f'PR for {class_name} (AP = {pr_auc[i]:.3f})',
                 color=colors[i % len(colors)], linewidth=2)
    
    # Plot the iso-f1 curves (curves of constant F1 score)
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(f'f1={f_score:0.1f}', xy=(0.9, y[45] + 0.02), color='gray')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    pr_path = os.path.join(output_dir, "pr_curves.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    
    logging.info(f"Precision-Recall curves saved to {pr_path}")
    return pr_auc

def print_prediction_summary(y_pred, y_true=None, output_dir=None):
    """Print a clear summary of predictions to console and append to summary file if output_dir is provided"""
    unique_preds, counts = np.unique(y_pred, return_counts=True)
    
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    # Table header
    header = f"{'Attack Type':<15} | {'Count':>8} | {'Percentage':>10}"
    if y_true is not None:
        header += f" | {'Accuracy':>12}"
    print(header)
    print("-" * len(header))
    
    # For appending to the summary file
    summary_lines = []
    summary_lines.append("\nPREDICTION SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    
    # Calculate and display stats for each attack type
    for pred, count in zip(unique_preds, counts):
        attack_name = ATTACK_TYPES.get(pred, f"Unknown ({pred})")
        percentage = (count / len(y_pred)) * 100
        
        # If ground truth is available, calculate accuracy for this class
        if y_true is not None:
            true_pos = np.sum((y_true == pred) & (y_pred == pred))
            actual_count = np.sum(y_true == pred)
            accuracy = true_pos / max(actual_count, 1) * 100
            line = f"{attack_name:<15} | {count:>8,d} | {percentage:>9.2f}% | {accuracy:>11.2f}%"
        else:
            line = f"{attack_name:<15} | {count:>8,d} | {percentage:>9.2f}%"
        
        print(line)
        summary_lines.append(line)
    
    print("-" * len(header))
    summary_lines.append("-" * len(header))
    
    # Append to summary file if output directory is provided
    if output_dir is not None:
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'a') as f:
            f.write("\n\n")  # Add some space before the prediction summary
            for line in summary_lines:
                f.write(line + "\n")

def add_random_value_noise(X, noise_fraction=0.2, random_state=None):
    """
    Replace a fraction of feature values with random values from the same feature.
    """
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    for j in range(n_features):
        mask = rng.random(n_samples) < noise_fraction
        if np.any(mask):
            random_values = rng.choice(X[:, j], size=np.sum(mask), replace=True)
            X_noisy[mask, j] = random_values
    return X_noisy

def main():
    root = Tk()
    root.withdraw()
    
    # Prompt user to select CSV file
    csv_file = askopenfilename(
        title="Select CSV file to test", 
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        parent=root
    )
    
    if not csv_file:
        print("No file selected. Exiting.")
        return
    
    # Print selected file to console
    print(f"\nSelected file: {csv_file}")
    
    # Configure logging BEFORE any logging calls
    configure_logging(csv_file)
    
    logging.info(f"Selected file: {csv_file}")
    logging.info(f"Testing model on: {csv_file}")
    
    try:
        # Load model and feature list
        print("Loading model...", end="")
        model, feature_list = load_model()
        print(" Done ✓")
        
        # Load scaler
        print("Loading data preprocessor...", end="")
        scaler = load_scaler()
        print(" Done ✓")
        
        # Load and preprocess data
        print("Loading and preprocessing data...", end="")
        X, y_true, df = load_and_preprocess_data(csv_file, feature_list)
        print(f" Done ✓  ({len(df):,d} rows)")
        logging.info(f"Data loaded and preprocessed. Shape: {X.shape}")
        
        # Create output directory based on input filename
        output_dir = create_output_directory(csv_file)
        
        # Add random value noise
        print("Adding random noise for robustness...", end="")
        X = add_random_value_noise(X, noise_fraction=0.2)
        print(" Done ✓")
        logging.info(f"Added random value noise to test data (20% of values replaced)")
        
        # Scale features
        print("Normalizing features...", end="")
        X_scaled = scaler.transform(X)
        print(" Done ✓")
        logging.info("Features scaled with StandardScaler")
        
        # Make predictions
        print("\nRunning predictions...", end="")
        y_pred, probabilities = predict(model, X_scaled)
        print(f" Done ✓  ({len(y_pred):,d} samples)")
        logging.info(f"Completed predictions for {len(y_pred):,d} samples")
        
        # Evaluate and visualize results
        print("Generating evaluation metrics and visualizations...", end="")
        evaluate_and_visualize(y_true, y_pred, probabilities, output_dir, df, csv_file, feature_list)
        print(" Done ✓")
        
        # Print prediction summary and add to summary file
        print_prediction_summary(y_pred, y_true, output_dir)
        
        # Clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Testing completed successfully!")
        logging.info(f"Results saved to: {os.path.abspath(output_dir)}")
        print(f"\nResults saved to: {os.path.abspath(output_dir)}")
        
        # Ask if user wants to test another file
        test_another = input("\nTest another file? (y/N): ").strip().lower() == "y"
        if test_another:
            main()
            
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        traceback.print_exc()
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
