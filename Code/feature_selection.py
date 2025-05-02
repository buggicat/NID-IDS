import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import seaborn as sns
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELECTION_FILE = os.path.join(PROJECT_ROOT, "Code", "selected_features.json")
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, "Code", "feature_weights.csv")

def select_optimal_features(df, target_column='attack', correlation_threshold=0.9, importance_threshold=0.01, min_correlation=0.05):
    """
    Automatically select optimal features for a model based on correlation, feature importance, and target correlation.

    Parameters:
    - df: Combined DataFrame containing the dataset.
    - target_column: Name of the target column (default: 'attack').
    - correlation_threshold: Threshold for removing highly correlated features (default: 0.9).
    - importance_threshold: Minimum importance score to keep a feature (default: 0.01).
    - min_correlation: Minimum correlation with the target to keep a feature (default: 0.05).

    Returns:
    - selected_features: List of selected features.
    - feature_importances: DataFrame with importance scores for all features.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Step 1: Correlation Analysis
    print("🔄 Performing correlation analysis...")
    correlation_matrix = X.corr()
    
    # Plot the correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    
    # Remove highly correlated features
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    
    print(f"⚠️ Removing {len(correlated_features)} highly correlated features...")
    X_uncorrelated = X.drop(columns=correlated_features)
    
    # Step 2: Remove features with very low correlation with the target
    print("🔄 Removing features with very low correlation with the target...")
    target_correlations = X_uncorrelated.corrwith(y)
    low_correlation_features = target_correlations[target_correlations.abs() < min_correlation].index.tolist()
    print(f"⚠️ Removing {len(low_correlation_features)} features with correlation < {min_correlation}...")
    X_filtered = X_uncorrelated.drop(columns=low_correlation_features)
    
    # Step 3: Feature Importance using Random Forest
    print("🔄 Calculating feature importance using Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_filtered, y)
    
    feature_importances = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    
    # Select features based on importance threshold
    selected_features = feature_importances[feature_importances['Importance'] >= importance_threshold]['Feature'].tolist()
    print(f"✅ Selected {len(selected_features)} features based on importance threshold ({importance_threshold}):")
    print(selected_features)
    
    return selected_features, feature_importances

def main():
    """Main function to select the input files and run feature selection."""
    # Create and hide the root window for file dialog
    root = Tk()
    root.withdraw()
    
    # Ask the user to select multiple CSV files
    csv_files = askopenfilenames(
        title="Select CSV files for feature selection",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not csv_files:
        print("No files selected. Exiting.")
        return
    
    print(f"Selected files: {csv_files}")
    
    # Combine all selected files into one DataFrame
    combined_df = pd.DataFrame()
    for file in csv_files:
        print(f"🔄 Loading file: {file}")
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    print(f"✅ Combined dataset shape: {combined_df.shape}")
    
    # Run the feature selection process
    selected_features, feature_importances = select_optimal_features(
        combined_df, 
        target_column='attack', 
        correlation_threshold=0.9, 
        importance_threshold=0.01, 
        min_correlation=0.05
    )
    
    # Save the selected features to a JSON file
    with open(SELECTION_FILE, "w") as f:
        json.dump(selected_features, f, indent=4)
    print(f"✅ Selected features saved to '{SELECTION_FILE}'")
    
    # Save all feature weights to a CSV file
    feature_importances.to_csv(WEIGHTS_FILE, index=False)
    print(f"✅ Feature weights saved to '{WEIGHTS_FILE}'")

if __name__ == "__main__":
    main()