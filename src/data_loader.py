import pandas as pd
import os

# Get absolute path to project root (one level above 'src')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_training_data(data_dir=DATA_DIR):
    """Load training data from CSV files."""
    train_file = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    df = pd.read_csv(train_file)
    
    # Verify required columns exist
    required_cols = ['id', ' efficiency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def load_test_data(data_dir=DATA_DIR):
    """Load test data from CSV files."""
    test_file = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    return pd.read_csv(test_file)

if __name__ == "__main__":
    try:
        train_df = load_training_data()
        print("Training data loaded successfully!")
        print(f"Shape: {train_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
        
        test_df = load_test_data()
        print("Test data loaded successfully!")
        print(f"Shape: {test_df.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
