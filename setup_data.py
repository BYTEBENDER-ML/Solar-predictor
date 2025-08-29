import zipfile
import os
import pandas as pd

def extract_and_setup_data():
    """Extract data from data.zip and set up the directory structure."""
    
    # Debug: show where we're looking
    print("Current working directory:", os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)
    print("Files in script directory:", os.listdir(script_dir))

    # Look for data.zip next to this script
    zip_path = os.path.join(script_dir, 'data.zip')
    print("Looking for:", zip_path)

    # Check if data.zip exists
    if not os.path.exists(zip_path):
        print("data.zip not found in the script directory")
        return False
    
    # Create data directory if it doesn't exist (place it next to script)
    data_dir = os.path.join(script_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Extract data
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Data extracted successfully to 'data/' directory")
        
        # List extracted files
        files = os.listdir(data_dir)
        print(f"Extracted files: {files}")
        
        # Check if train.csv and test.csv exist
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        
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
