#!/usr/bin/env python3
"""
Script to create a zip file of the BIODD project with all necessary components.
Run this script from the project's root directory.
"""

import os
import zipfile
import sys

def create_project_zip(output_name='project3.zip'):
    # Files to include
    essential_files = [
        # Python scripts
        '1.Download_Bioactivity_Data.py',
        '2.Exploratory_Data_Analysis.py',
        '3.Descriptor_Calculation_and_Dataset_Preparation.py',
        '4.Regression_Models_with_Random_Forest.py',
        '5.Comparing_Regressors.py',
        
        # Configuration and documentation
        'requirements.txt',
        'README.md',
        
        # Supporting files
        'molecule.smi',
        'target_information.csv',
    ]
    
    # Directories to include
    directories = [
        'data',
        'demo_data',
        'descriptors',
        '3d_structures',
        'pharmacophores',
        'plots',
        'models',
        'results',
        'target_datasets',
    ]
    
    # Check if essential files exist
    missing_files = [f for f in essential_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: The following files are missing: {', '.join(missing_files)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create zip file
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add essential files
        for file in essential_files:
            if os.path.exists(file):
                print(f"Adding file: {file}")
                zipf.write(file)
        
        # Add directories
        for directory in directories:
            if os.path.exists(directory):
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"Adding file: {file_path}")
                        zipf.write(file_path)
            else:
                print(f"Directory not found: {directory}, skipping...")
    
    print(f"\nZip file created: {output_name}")
    print(f"Size: {os.path.getsize(output_name) / (1024*1024):.2f} MB")
    print("\nInstructions:")
    print("1. Upload this zip file to your Google VM")
    print("2. Unzip it using 'unzip biodd_project_vm.zip'")
    print("3. Follow the instructions in README.md")

if __name__ == "__main__":
    create_project_zip() 