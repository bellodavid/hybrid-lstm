#!/usr/bin/env python3
"""
Setup script to copy model files to the app directory
"""

import os
import shutil

def setup_model_files():
    """Copy model files from parent directory to app directory"""
    
    # Define paths
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    
    # Files to copy
    files_to_copy = [
        'aflatoxin_model.onnx',
        'standard_scaler_params.json', 
        'label_encoder_classes.json'
    ]
    
    print(f"Looking for model files in: {parent_dir}")
    print(f"Copying to: {app_dir}")
    
    for filename in files_to_copy:
        src_path = os.path.join(parent_dir, filename)
        dst_path = os.path.join(app_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {filename}")
        else:
            print(f"✗ {filename} not found in {parent_dir}")
    
    print("Model setup complete!")

if __name__ == "__main__":
    setup_model_files()