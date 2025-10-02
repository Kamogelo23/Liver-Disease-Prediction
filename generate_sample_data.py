#!/usr/bin/env python3
"""
Sample Dataset Generator for Liver Disease Prediction
Creates a realistic HCV dataset for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_realistic_hcv_dataset(n_samples=615, save_path="hcvdat0.csv"):
    """
    Create a realistic HCV dataset based on medical literature and typical values.
    
    Args:
        n_samples (int): Number of samples to generate
        save_path (str): Path to save the dataset
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    np.random.seed(42)
    
    # Define realistic ranges for each biomarker based on medical literature
    biomarker_ranges = {
        'Age': {'normal': (18, 80), 'cirrhosis': (40, 75), 'hepatitis': (20, 60)},
        'ALB': {'normal': (3.5, 5.0), 'cirrhosis': (2.0, 3.5), 'hepatitis': (3.0, 4.5)},  # Albumin (g/dL)
        'ALP': {'normal': (44, 147), 'cirrhosis': (100, 400), 'hepatitis': (60, 200)},    # Alkaline Phosphatase (U/L)
        'ALT': {'normal': (7, 56), 'cirrhosis': (30, 200), 'hepatitis': (50, 500)},      # Alanine Aminotransferase (U/L)
        'AST': {'normal': (10, 40), 'cirrhosis': (40, 300), 'hepatitis': (30, 400)},     # Aspartate Aminotransferase (U/L)
        'BIL': {'normal': (0.3, 1.2), 'cirrhosis': (1.0, 5.0), 'hepatitis': (0.5, 3.0)}, # Bilirubin (mg/dL)
        'CHE': {'normal': (5.5, 12.9), 'cirrhosis': (2.0, 8.0), 'hepatitis': (4.0, 10.0)}, # Cholinesterase (kU/L)
        'CHOL': {'normal': (120, 200), 'cirrhosis': (100, 180), 'hepatitis': (120, 220)}, # Cholesterol (mg/dL)
        'CREA': {'normal': (0.6, 1.2), 'cirrhosis': (0.8, 2.0), 'hepatitis': (0.6, 1.5)}, # Creatinine (mg/dL)
        'GGT': {'normal': (8, 61), 'cirrhosis': (50, 500), 'hepatitis': (30, 300)},       # Gamma-glutamyl transferase (U/L)
        'PROT': {'normal': (6.0, 8.3), 'cirrhosis': (5.0, 7.5), 'hepatitis': (6.0, 8.0)}  # Protein (g/dL)
    }
    
    # Generate data for each category
    categories = ['Blood donor', 'Cirrhosis', 'Fibrosis', 'Hepatitis']
    category_probs = [0.35, 0.25, 0.20, 0.20]  # Realistic distribution
    
    data = []
    
    for category in categories:
        n_category = int(n_samples * category_probs[categories.index(category)])
        
        for i in range(n_category):
            sample = {}
            
            # Age (varies by category)
            if category == 'Blood donor':
                sample['Age'] = np.random.normal(35, 12)
            elif category == 'Cirrhosis':
                sample['Age'] = np.random.normal(55, 10)
            elif category == 'Fibrosis':
                sample['Age'] = np.random.normal(50, 12)
            else:  # Hepatitis
                sample['Age'] = np.random.normal(40, 15)
            
            sample['Age'] = max(18, min(80, int(sample['Age'])))
            
            # Sex (slightly more males in liver disease)
            if category == 'Blood donor':
                sample['Sex'] = np.random.choice(['m', 'f'], p=[0.55, 0.45])
            else:
                sample['Sex'] = np.random.choice(['m', 'f'], p=[0.65, 0.35])
            
            # Generate biomarkers based on category
            for biomarker, ranges in biomarker_ranges.items():
                if biomarker == 'Age':
                    continue
                    
                # Select range based on category
                if category == 'Blood donor':
                    range_key = 'normal'
                elif category == 'Cirrhosis':
                    range_key = 'cirrhosis'
                elif category == 'Fibrosis':
                    # Fibrosis is intermediate between normal and cirrhosis
                    range_key = np.random.choice(['normal', 'cirrhosis'], p=[0.6, 0.4])
                else:  # Hepatitis
                    range_key = 'hepatitis'
                
                min_val, max_val = ranges[range_key]
                
                # Generate value with some variability
                if biomarker in ['ALB', 'BIL', 'CHE', 'CREA', 'PROT']:
                    # Continuous values with 2 decimal places
                    value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
                    sample[biomarker] = round(max(min_val, min(max_val, value)), 2)
                else:
                    # Integer values
                    value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
                    sample[biomarker] = round(max(min_val, min(max_val, value)))
            
            sample['Category'] = category
            data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some realistic missing values (about 5% missing)
    missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    missing_features = np.random.choice(['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE'], 
                                      size=len(missing_indices), replace=True)
    
    for idx, feature in zip(missing_indices, missing_features):
        df.loc[idx, feature] = np.nan
    
    # Ensure we have exactly n_samples
    if len(df) > n_samples:
        df = df.head(n_samples)
    elif len(df) < n_samples:
        # Add more samples by duplicating and adding noise
        additional_needed = n_samples - len(df)
        additional_samples = df.sample(n=additional_needed, replace=True, random_state=42)
        # Add small random noise to avoid exact duplicates
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'Age':
                noise = np.random.normal(0, 0.1, len(additional_samples))
                additional_samples[col] += noise
        df = pd.concat([df, additional_samples], ignore_index=True)
    
    # Shuffle again
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    print(f"Dataset created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {save_path}")
    print(f"\nCategory distribution:")
    print(df['Category'].value_counts())
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df

def create_sample_dataset_small(n_samples=100, save_path="sample_hcv_data.csv"):
    """
    Create a smaller sample dataset for quick testing.
    """
    return create_realistic_hcv_dataset(n_samples, save_path)

if __name__ == "__main__":
    # Create the main dataset
    print("Creating realistic HCV dataset...")
    df = create_realistic_hcv_dataset(n_samples=615, save_path="hcvdat0.csv")
    
    # Create a smaller sample for quick testing
    print("\nCreating small sample dataset...")
    df_small = create_sample_dataset_small(n_samples=100, save_path="sample_hcv_data.csv")
    
    print("\nDataset generation completed!")
    print("Files created:")
    print("- hcvdat0.csv (main dataset)")
    print("- sample_hcv_data.csv (small sample)")
