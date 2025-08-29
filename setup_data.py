import zipfile
import os
import pandas as pd

def extract_and_setup_data():
    """Extract data from data.zip and set up the directory structure."""
    
    # Check if data.zip exists
    if not os.path.exists('data.zip'):
        print("data.zip not found in the current directory")
        return False
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Extract data
    try:
        with zipfile.ZipFile('data.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
        print("Data extracted successfully to 'data/' directory")
        
        # List extracted files
        files = os.listdir('data')
        print(f"Extracted files: {files}")
        
        # Check if train.csv and test.csv exist
        train_path = os.path.join('data', 'train.csv')
        test_path = os.path.join('data', 'test.csv')
        
        if os.path.exists(train_path):
            # Load and check train data
            train_df = pd.read_csv(train_path)
            print(f"Train data shape: {train_df.shape}")
            print(f"Train columns: {train_df.columns.tolist()}")
            
            # Check for required columns
            required_cols = ['id', 'efficiency']
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns in train.csv: {missing_cols}")
            else:
                print("✓ All required columns found in train.csv")
        
        if os.path.exists(test_path):
            # Load and check test data
            test_df = pd.read_csv(test_path)
            print(f"Test data shape: {test_df.shape}")
            print(f"Test columns: {test_df.columns.tolist()}")
            
            # Check for required columns (test data might not have 'efficiency')
            if 'id' in test_df.columns:
                print("✓ Test data has 'id' column")
            else:
                print("Warning: Test data missing 'id' column")
        
        return True
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return False

if __name__ == "__main__":
    extract_and_setup_data()
