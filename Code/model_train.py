# Standard library imports
import os
import sys
import json
import pickle
import logging
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# GUI imports - only imported when needed
def get_file_dialog():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename, askopenfilenames
    return Tk(), askopenfilename, askopenfilenames

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIRS = {
    "model": os.path.join(PROJECT_ROOT, "trained_model"),
    "logs": os.path.join(PROJECT_ROOT, "Logs"),
    "results": os.path.join(PROJECT_ROOT, "Results")
}
FEATURE_FILE = os.path.join(PROJECT_ROOT, "Code", "feature_dict.json")

# Create directories if they don't exist
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Update global references to match the dictionary
MODEL_DIR = DIRS["model"]
LOG_DIR = DIRS["logs"]
RESULTS_DIR = DIRS["results"]

# Attack type mapping
ATTACK_TYPES = {
    0: 'Normal',
    1: 'DDoS',
    2: 'Memcrashed',
    3: 'PortScan'
}

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure logging
def setup_logging(mode="train"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{mode}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Network architecture - more focused for IDS
class IDSNet(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(IDSNet, self).__init__()
        
        # Attention layer
        self.attention = nn.Linear(input_size, input_size)  # Learn feature-wise weights
        self.attention_softmax = nn.Softmax(dim=1)  # Normalize weights
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16) 
        self.fc_out = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.7) 
    
    def forward(self, x):
        # Attention mechanism
        attention_weights = self.attention_softmax(self.attention(x))  # Compute attention weights
        x = x * attention_weights  # Apply attention weights to input features
        
        # Pass through the rest of the network
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# Data loading and preprocessing
def load_data(file_path, feature_list=None):
    """Load and preprocess a CSV file"""
    logging.info(f"Loading data from {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Ensure the 'attack' column is preserved
        if 'attack' not in df.columns:
            raise ValueError("The dataset must contain an 'attack' column for ground truth labels.")
        
        # Check if all required features are present
        if feature_list:
            missing_features = [f for f in feature_list if f not in df.columns]
            if missing_features:
                logging.info(f"Adding missing features to dataset: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0
            
            # Ensure the 'attack' column is included in the filtered dataset
            required_columns = feature_list + ['attack']
            df = df[[col for col in required_columns if col in df.columns]]
        
        # Convert all numeric columns
        numeric_cols = [col for col in df.columns if col != 'attack']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df.fillna(0, inplace=True)
        
        # Extract features and labels
        X = df[feature_list].values
        y = df['attack'].values.astype(int)
        
        return X, y, df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def prepare_data_for_training(X, y, test_size=0.2, random_state=42):
    """Prepare data for model training with proper scaling"""
    logging.info("Preparing data for training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")
    
    # Count classes in training data
    class_counts = np.bincount(y_train)
    for class_idx, count in enumerate(class_counts):
        class_name = ATTACK_TYPES.get(class_idx, f"Class {class_idx}")
        logging.info(f"Class {class_name}: {count} samples in training")
    
    # Create PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler

def add_feature_noise(X, y, noise_level=0.05):
    """
    Add uniform random Gaussian noise to features.
    
    Parameters:
    - X: Feature matrix (numpy array)
    - y: Labels array (not used in this implementation)
    - noise_level: Proportion of the feature's standard deviation to use as noise
    
    Returns:
    - X_noisy: Feature matrix with added noise
    """
    X_noisy = X.copy()
    
    # Standard uniform noise
    noise = np.random.normal(0, noise_level * X.std(axis=0), X.shape)
    X_noisy = X + noise
    
    return X_noisy

def add_label_noise(y, noise_rate=0.05):
    """
    Add uniform noise to labels by randomly flipping a percentage of them.
    
    Parameters:
    - y: Label array (numpy array)
    - noise_rate: Proportion of labels to flip
    
    Returns:
    - y_noisy: Label array with noise added
    """
    y_noisy = y.copy()
    unique_labels = np.unique(y)
    
    # Calculate number of labels to flip
    num_noisy = int(len(y) * noise_rate)
    
    # Randomly select indices to flip
    if num_noisy > 0:
        noisy_indices = np.random.choice(len(y), size=num_noisy, replace=False)
        
        for idx in noisy_indices:
            current_label = y_noisy[idx]
            # Assign a different label randomly
            new_label = np.random.choice(unique_labels[unique_labels != current_label])
            y_noisy[idx] = new_label
    
    return y_noisy

def prepare_data_for_training_with_noise(X, y, train_size=0.7, val_size=0.1, test_size=0.2, 
                                        random_state=42, noise_level=0.05, label_noise_rate=0.05,
                                        balance_strategy="combine"):
    """
    Prepare data for model training with dataset balancing, added noise, and proper scaling.
    
    Parameters:
    - X: Feature matrix (numpy array)
    - y: Label array (numpy array)
    - train_size: Proportion of data to use for training 
    - val_size: Proportion of data to use for validation
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    - noise_level: Proportion of feature noise to add
    - label_noise_rate: Proportion of label noise to add
    - balance_strategy: Strategy for balancing the dataset
    
    Returns:
    - X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor, scaler
    """
    logging.info("Preparing data for training with dataset balancing and uniform noise injection")
    logging.info(f"Data split: Train {train_size*100:.0f}%, Validation {val_size*100:.0f}%, Test {test_size*100:.0f}%")
    
    # Step 1: Balance the dataset
    logging.info(f"Balancing dataset using strategy: {balance_strategy}")
    X_balanced, y_balanced = balance_classes(X, y, strategy=balance_strategy)
    logging.info("Dataset balanced successfully")
    
    # Step 2: Add feature noise
    logging.info(f"Adding uniform feature noise, level: {noise_level}")
    X_noisy = add_feature_noise(X_balanced, y_balanced, noise_level=noise_level)
    
    # Step 3: Add label noise
    logging.info(f"Adding uniform label noise, rate: {label_noise_rate}")
    y_noisy = add_label_noise(y_balanced, noise_rate=label_noise_rate)
    
    # Step 4: First split into training and temp (validation + testing)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_noisy, y_noisy, test_size=(val_size + test_size), random_state=random_state, stratify=y_balanced
    )
    
    # Step 5: Split the temp set into validation and testing
    # Adjust test_size to get the right proportion from the temp set
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    
    # Step 6: Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")
    
    # Log class distribution in each set
    for name, labels in [("Training", y_train), ("Validation", y_val), ("Testing", y_test)]:
        class_counts = np.bincount(labels)
        for class_idx, count in enumerate(class_counts):
            if class_idx < len(ATTACK_TYPES):
                class_name = ATTACK_TYPES.get(class_idx, f"Class {class_idx}")
                logging.info(f"{name} set - Class {class_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Create PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor, scaler

def create_weighted_sampler(y_train):
    """Create a weighted sampler to handle class imbalance."""
    class_counts = np.bincount(y_train.numpy())
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[y_train]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )
    
    return sampler

def balance_classes(X, y, strategy="undersample"):
    """
    Balance the dataset by undersampling or oversampling.

    Parameters:
    - X: Features (numpy array or pandas DataFrame)
    - y: Labels (numpy array)
    - strategy: Balancing strategy ("undersample", "oversample", or "combine")

    Returns:
    - X_balanced: Balanced features
    - y_balanced: Balanced labels
    """
    from sklearn.utils import resample
    import pandas as pd
    import numpy as np

    # Combine features and labels into a single DataFrame for easier processing
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    data['label'] = y

    # Analyze class distribution
    class_counts = data['label'].value_counts()
    logging.info(f"Original class distribution:\n{class_counts}")

    # Determine the target number of samples per class
    if strategy == "undersample":
        target_count = class_counts.min()  # Match the minority class
    elif strategy == "oversample":
        target_count = class_counts.max()  # Match the majority class
    elif strategy == "combine":
        target_count = int(class_counts.mean())  # Average class size
    else:
        raise ValueError("Invalid strategy. Choose from 'undersample', 'oversample', or 'combine'.")

    # Balance the dataset
    balanced_data = []
    for label in class_counts.index:
        class_data = data[data['label'] == label]

        if strategy == "undersample":
            class_data = class_data.sample(target_count, random_state=42)
        elif strategy == "oversample":
            class_data = resample(class_data, replace=True, n_samples=target_count, random_state=42)
        elif strategy == "combine":
            if len(class_data) > target_count:
                class_data = class_data.sample(target_count, random_state=42)
            else:
                class_data = resample(class_data, replace=True, n_samples=target_count, random_state=42)

        balanced_data.append(class_data)

    # Combine all balanced classes
    balanced_df = pd.concat(balanced_data)

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split features and labels
    X_balanced = balanced_df.iloc[:, :-1].values
    y_balanced = balanced_df.iloc[:, -1].values.astype(int)

    # Log the new class distribution
    new_class_counts = pd.Series(y_balanced).value_counts()
    logging.info(f"Balanced class distribution:\n{new_class_counts}")

    return X_balanced, y_balanced

def calibrate_prediction_thresholds(model, X_val, y_val, batch_size=256):
    """
    Calibrate optimal prediction thresholds for each class to improve model performance.
    
    Parameters:
    - model: Trained model
    - X_val: Validation features (tensor)
    - y_val: Validation labels (tensor)
    - batch_size: Batch size for processing
    
    Returns:
    - thresholds: List of calibrated thresholds for each class
    """
    logging.info("Calibrating prediction thresholds based on validation data")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Get all predictions and probabilities
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine batches
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Number of classes
    n_classes = len(ATTACK_TYPES)
    
    # Find optimal thresholds using ROC curve
    thresholds = []
    for i in range(n_classes):
        # Create binary target (class i vs rest)
        binary_targets = (all_targets == i).astype(int)
        
        # Get probabilities for this class
        class_probs = all_probs[:, i]
        
        # Find threshold that maximizes F1-score
        best_f1 = 0
        best_threshold = 0.5  # Default threshold
        
        for threshold in np.arange(0.1, 1.0, 0.05):
            preds = (class_probs >= threshold).astype(int)
            f1 = np.sum(preds & binary_targets) * 2 / (np.sum(preds) + np.sum(binary_targets))
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds.append(best_threshold)
        logging.info(f"Class {ATTACK_TYPES[i]}: calibrated threshold = {best_threshold:.2f}")
    
    logging.info(f"Calibrated thresholds: {thresholds}")
    return thresholds

def train_model(model, X_train, y_train, X_val, y_val, 
                num_epochs=3, batch_size=128, learning_rate=0.001, weight_decay=1e-4):
    """Train the IDS model with early stopping and L2 regularization."""
    logging.info("Starting model training with WeightedRandomSampler and L2 regularization")
    
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights based on the training data
    class_counts = np.bincount(y_train.numpy())
    total_samples = len(y_train)

    # Adjust weights to balance the classes
    class_weights = [
        total_samples / (len(class_counts) * count) if count > 0 else 0
        for count in class_counts
    ]

    # Convert to tensor and move to the same device as the model
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Define the loss function with updated class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Define optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Create WeightedRandomSampler
    train_sampler = create_weighted_sampler(y_train)
    
    # Define DataLoader with sampler
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    # Validation DataLoader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 
        'best_val_acc': 0.0
    }
    
    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            logging.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logging.info(f"Early stopping counter: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load the best model before finishing
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt")))
    logging.info("Loaded best model for final evaluation")
    
    return model, history

def print_prediction_summary(y_pred, y_true=None, output_dir=None):
    """Print a clear summary of predictions to console and append to model summary file if output_dir is provided"""
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
    
    # Append to model summary file if output directory is provided
    if output_dir is not None:
        model_summary_path = os.path.join(output_dir, "model_summary.txt")
        with open(model_summary_path, 'a') as f:
            f.write("\n\n")  # Add some space before the prediction summary
            for line in summary_lines:
                f.write(line + "\n")

def evaluate_model(model, X_data, y_data, batch_size=128, dataset_type="test", visualize=False, thresholds=None, results_dir=None):
    """
    Evaluate model on provided data
    
    Parameters:
    - model: Trained model to evaluate
    - X_data: Feature data (tensor)
    - y_data: Ground truth labels (tensor)
    - batch_size: Batch size for evaluation
    - dataset_type: 'train', 'validation' or 'test' (for logging purposes)
    - visualize: Whether to create individual visualizations
    - thresholds: Optional calibration thresholds
    - results_dir: Directory to save results (if None, a new directory will be created)
    
    Returns:
    - accuracy, report, confusion_matrix, y_pred, y_true
    """
    logging.info(f"Evaluating model on {dataset_type} data")
    
    model = model.to(device)
    model.eval()
    
    # Create data loader
    dataset = TensorDataset(X_data, y_data)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use calibrated prediction if thresholds are provided
            if thresholds:
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                # Convert the boolean tensor to a float tensor before applying argmax
                comparison = (probabilities > torch.tensor(thresholds).to(device)).float()
                predicted = torch.argmax(comparison, dim=1)
            else:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"{dataset_type.capitalize()} Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=[ATTACK_TYPES.get(i, f"Class {i}") for i in range(len(ATTACK_TYPES))],
        output_dict=True
    )
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Return all five values as expected by the calling code
    return accuracy, report, cm, y_pred, y_true

def display_classification_results(val_report, test_report):
    """
    Display classification results for validation and test datasets side by side.
    
    Parameters:
    - val_report: Validation classification report (dict)
    - test_report: Test classification report (dict)
    """
    print("\n" + "="*100)
    print("CLASSIFICATION RESULTS COMPARISON")
    print("="*100)
    
    # Print overall accuracy
    print(f"\nOverall Accuracy:")
    print(f"  Validation: {val_report['accuracy']:.4f}")
    print(f"  Test:       {test_report['accuracy']:.4f}")
    print(f"  Difference: {test_report['accuracy'] - val_report['accuracy']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Performance Metrics:")
    print("-"*100)
    print(f"{'Class':<12} | {'Validation':<40} | {'Test':<40}")
    print(f"{'':12} | {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8} | " + 
          f"{'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print("-"*100)
    
    for class_label in sorted([c for c in val_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]):
        val_metrics = val_report[class_label]
        test_metrics = test_report[class_label]
        class_name = ATTACK_TYPES.get(int(class_label), f"Class {class_label}") if class_label.isdigit() else class_label
        
        print(f"{class_name:<12} | " + 
              f"{val_metrics['precision']:>10.4f} {val_metrics['recall']:>10.4f} " + 
              f"{val_metrics['f1-score']:>10.4f} {val_metrics['support']:>8} | " +
              f"{test_metrics['precision']:>10.4f} {test_metrics['recall']:>10.4f} " + 
              f"{test_metrics['f1-score']:>10.4f} {test_metrics['support']:>8}")
    
    print("-"*100)
    
    # Print average metrics
    for avg_type in ['macro avg', 'weighted avg']:
        val_metrics = val_report[avg_type]
        test_metrics = test_report[avg_type]
        
        print(f"{avg_type:<12} | " + 
              f"{val_metrics['precision']:>10.4f} {val_metrics['recall']:>10.4f} " + 
              f"{val_metrics['f1-score']:>10.4f} {val_metrics['support']:>8} | " +
              f"{test_metrics['precision']:>10.4f} {test_metrics['recall']:>10.4f} " + 
              f"{test_metrics['f1-score']:>10.4f} {test_metrics['support']:>8}")
    
    print("="*100)

def create_roc_pr_curves(model, X_data, y_data, dataset_type, results_dir):
    """
    Create and save ROC and PR curves for multi-class classification.
    
    Parameters:
    - model: Trained model
    - X_data: Feature data (tensor)
    - y_data: Ground truth labels (tensor)
    - dataset_type: Name of the dataset ('validation' or 'test')
    - results_dir: Directory to save the curves
    """
    try:
        logging.info(f"Creating ROC and PR curves for {dataset_type} data")
        
        model = model.to(device)
        model.eval()
        
        # Get class probabilities
        y_true = y_data.cpu().numpy()
        
        # Create a one-hot encoded version of the target
        n_classes = len(ATTACK_TYPES)
        y_onehot = np.zeros((y_true.size, n_classes))
        for i in range(n_classes):
            y_onehot[:, i] = (y_true == i).astype(int)
        
        # Get probabilities for each class
        dataset = TensorDataset(X_data)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        y_probs = []
        
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                y_probs.append(probs.cpu().numpy())
        
        y_probs = np.vstack(y_probs)
        
        # Import metrics functions
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        # Create plots
        # 1. ROC Curve for each class
        plt.figure(figsize=(12, 10))
        
        # Plot ROC curves
        for i, class_name in ATTACK_TYPES.items():
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                     label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {dataset_type.capitalize()} Data')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(results_dir, f"{dataset_type}_roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curve for each class
        plt.figure(figsize=(12, 10))
        
        for i, class_name in ATTACK_TYPES.items():
            precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_probs[:, i])
            avg_precision = average_precision_score(y_onehot[:, i], y_probs[:, i])
            plt.plot(recall, precision, lw=2,
                     label=f'{class_name} (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves for {dataset_type.capitalize()} Data')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        pr_path = os.path.join(results_dir, f"{dataset_type}_pr_curves.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved ROC and PR curves for {dataset_type} data to {results_dir}")
        
        return roc_path, pr_path
    
    except Exception as e:
        logging.error(f"Error creating curves for {dataset_type} data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def train_mode():
    """Run training process to create a robust IDS model with noise injection."""
    setup_logging("train")
    logging.info("Starting training process with noise injection")
    
    # Load feature list
    with open(FEATURE_FILE, 'r') as f:
        feature_list = json.load(f)
    
    # Prompt user to select multiple CSV files for different attack types
    combined_data = None
    combined_labels = None
    
    while True:
        # Use multiple file selection directly
        root, askopenfilename, askopenfilenames = get_file_dialog()
        root.withdraw()
        csv_files = askopenfilenames(
            title="Select CSV files containing attack data (you can select multiple files)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=root
        )
        
        if not csv_files:
            break
            
        file_list = list(csv_files)  # Convert tuple to list
        logging.info(f"Selected {len(file_list)} files for processing")
        
        # Process each file in the list
        for csv_file in file_list:
            try:
                logging.info(f"Processing file: {csv_file}")
                X, y, _ = load_data(csv_file, feature_list)
                
                # Check if labels are present
                if y is None:
                    logging.error(f"No 'attack' column found in {csv_file}. Skipping file.")
                    continue
                
                # Append to combined dataset
                if combined_data is None:
                    combined_data = X
                    combined_labels = y
                else:
                    combined_data = np.vstack((combined_data, X))
                    combined_labels = np.append(combined_labels, y)
                
                # Show class distribution in this file
                unique_labels, label_counts = np.unique(y, return_counts=True)
                logging.info(f"Added {len(y)} samples from {csv_file}")
                for label, count in zip(unique_labels, label_counts):
                    label_name = ATTACK_TYPES.get(label, f"Class {label}")
                    logging.info(f"  {label_name}: {count} samples")
                
            except Exception as e:
                logging.error(f"Error processing {csv_file}: {str(e)}")
        
        # Ask if user wants to add more files
        add_more = input("Add more data files? (y/n): ").lower() == 'y'
        if not add_more:
            break
    
    if combined_data is None or len(combined_data) == 0:
        logging.error("No valid data loaded. Exiting.")
        return
    
    # Log final dataset stats
    unique_labels, label_counts = np.unique(combined_labels, return_counts=True)
    logging.info(f"Final dataset: {len(combined_labels)} total samples")
    for label, count in zip(unique_labels, label_counts):
        label_name = ATTACK_TYPES.get(label, f"Class {label}")
        logging.info(f"  {label_name}: {count} samples ({count/len(combined_labels)*100:.1f}%)")
    
    # Create a single parent results folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_results_dir = os.path.join(RESULTS_DIR, f"model_evaluation_{timestamp}")
    os.makedirs(parent_results_dir, exist_ok=True)
    logging.info(f"Created main results directory: {parent_results_dir}")
    
    # Prepare data for training with noise injection
    logging.info("Preparing data for training with noise injection...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_for_training_with_noise(
        combined_data, combined_labels, 
        train_size=0.7, val_size=0.1, test_size=0.2,
        noise_level=0.10,
        label_noise_rate=0.05,
        balance_strategy="combine"
    )
    
    # Create model
    model = IDSNet(input_size=len(feature_list))
    
    # Train model
    model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        num_epochs=3, batch_size=128
    )
    
    # Calibrate prediction thresholds
    thresholds = calibrate_prediction_thresholds(model, X_val, y_val)
    
    # Evaluate model on validation set with calibrated thresholds
    val_accuracy, val_report, val_cm, val_pred, val_true = evaluate_model(
        model, X_val, y_val, dataset_type="validation", 
        thresholds=thresholds, visualize=False, 
        results_dir=os.path.join(parent_results_dir, "validation_results")
    )
    
    # Evaluate model on test set with calibrated thresholds
    test_accuracy, test_report, test_cm, test_pred, test_true = evaluate_model(
        model, X_test, y_test, dataset_type="test",
        thresholds=thresholds, visualize=False, 
        results_dir=os.path.join(parent_results_dir, "test_results")
    )
    
    # Save model summary and parameters in the parent folder with added comparison
    logging.info("Writing detailed summary with comparison data")
    with open(os.path.join(parent_results_dir, "model_summary.txt"), 'w') as f:
        f.write(f"Model trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Validation samples: {len(y_val)}\n")
        f.write(f"Test samples: {len(y_test)}\n\n")
        f.write(f"Final validation accuracy: {val_accuracy:.4f}\n")
        f.write(f"Final test accuracy: {test_accuracy:.4f}\n")
        f.write(f"Noise parameters:\n")
        f.write(f"  - Feature noise level: 0.10\n")
        f.write(f"  - Label noise rate: 0.05\n")
        f.write(f"  - Feature noise strategy: uniform\n")
        f.write(f"  - Label noise strategy: uniform\n\n")
        
        # Add detailed comparison section
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION RESULTS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Accuracy:\n")
        f.write(f"  Validation: {val_report['accuracy']:.4f}\n")
        f.write(f"  Test:       {test_report['accuracy']:.4f}\n")
        f.write(f"  Difference: {test_report['accuracy'] - val_report['accuracy']:.4f}\n\n")
        
        f.write("Per-Class Performance Metrics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Class':<12} | {'Validation':<40} | {'Test':<40}\n")
        f.write(f"{'':12} | {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8} | " + 
                f"{'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}\n")
        f.write("-" * 100 + "\n")
        
        for class_label in sorted([c for c in val_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]):
            val_metrics = val_report[class_label]
            test_metrics = test_report[class_label]
            class_name = ATTACK_TYPES.get(int(class_label), f"Class {class_label}") if class_label.isdigit() else class_label
            
            f.write(f"{class_name:<12} | " + 
                    f"{val_metrics['precision']:>10.4f} {val_metrics['recall']:>10.4f} " + 
                    f"{val_metrics['f1-score']:>10.4f} {val_metrics['support']:>8} | " +
                    f"{test_metrics['precision']:>10.4f} {test_metrics['recall']:>10.4f} " + 
                    f"{test_metrics['f1-score']:>10.4f} {test_metrics['support']:>8}\n")
        
        f.write("-" * 100 + "\n")
        
        for avg_type in ['macro avg', 'weighted avg']:
            val_metrics = val_report[avg_type]
            test_metrics = test_report[avg_type]
            
            f.write(f"{avg_type:<12} | " + 
                    f"{val_metrics['precision']:>10.4f} {val_metrics['recall']:>10.4f} " + 
                    f"{val_metrics['f1-score']:>10.4f} {val_metrics['support']:>8} | " +
                    f"{test_metrics['precision']:>10.4f} {test_metrics['recall']:>10.4f} " + 
                    f"{test_metrics['f1-score']:>10.4f} {test_metrics['support']:>8}\n")
        
        f.write("=" * 100 + "\n")
    
    # Print validation dataset prediction summary and append to model_summary.txt
    print("\nVALIDATION DATASET")
    print_prediction_summary(val_pred, val_true, parent_results_dir)
    
    # Print test dataset prediction summary and append to model_summary.txt
    print("\nTEST DATASET")
    print_prediction_summary(test_pred, test_true, parent_results_dir)
    
    # Add ROC and PR curves for both validation and test sets
    logging.info("Generating ROC and PR curves for validation and test data")
    val_roc_path, val_pr_path = create_roc_pr_curves(model, X_val, y_val, "validation", parent_results_dir)
    test_roc_path, test_pr_path = create_roc_pr_curves(model, X_test, y_test, "test", parent_results_dir)
    
    if val_roc_path and test_roc_path:
        logging.info("Successfully created ROC and PR curves")
    else:
        logging.warning("Failed to create some or all curves")
    
    # Display classification results
    display_classification_results(val_report, test_report)
    
    logging.info(f"All evaluation results saved to: {parent_results_dir}")
    logging.info(f"Final validation accuracy: {val_accuracy:.4f}")
    logging.info(f"Final test accuracy: {test_accuracy:.4f}")
    logging.info("Training process completed successfully")
    
    return model

def main():
    """Main entry point for the IDS training application"""
    print("=" * 80)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 80)
    print("\nThis application will train a model on multiple attack datasets.")
    print("You'll be prompted to select CSV files containing different attack types.")
    print("\nThe trained model will be saved for later use with the testing application.")
    
    train_mode()
    
    print("\nTraining completed. Run model_test.py to test the model on new data.")

if __name__ == "__main__":
    main()

