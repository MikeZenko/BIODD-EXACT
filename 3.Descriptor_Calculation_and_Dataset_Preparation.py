#Load bioactivity data and calculate PubChem fingerprints
import pandas as pd
import numpy as np
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf, GraphDescriptors

print("Step 3: Descriptor Calculation and Dataset Preparation")

# Load the preprocessed dataset from Step 2
input_file = 'pancreatic_cancer_bioactivity_data_2class_lipinski.csv'
if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} not found! Run Step 2 first.")
    sys.exit(1)
    
df3 = pd.read_csv(input_file)
print(f"Loaded data with {len(df3)} compounds and {df3.shape[1]} features")

# Extract SMILES and ChEMBL IDs for descriptor calculation
selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
print("Created SMILES file for descriptor calculation")

# Show a sample of the SMILES file
with open('molecule.smi', 'r') as f:
    sample = [next(f) for _ in range(min(5, len(df3)))]
print("SMILES file sample (first 5 lines):")
for line in sample:
    print(line.strip())

print(f"Total molecules for descriptor calculation: {len(df3)}")

# Calculate multiple descriptor types using RDKit
print("Calculating molecular descriptors...")

# Lists to store fingerprint and descriptor data
fingerprint_data = []
failed_mols = 0

# Morgan fingerprint parameters
radius = 2  # ECFP4
nBits = 1024  # Number of bits

# Dictionary to map physicochemical descriptor names to their calculation functions
physchem_descriptors = {
    # Basic properties
    'MolWt': Descriptors.MolWt,
    'LogP': Descriptors.MolLogP,
    'NumHDonors': Lipinski.NumHDonors,
    'NumHAcceptors': Lipinski.NumHAcceptors,
    'NumRotatableBonds': Lipinski.NumRotatableBonds,
    'NumAromaticRings': Lipinski.NumAromaticRings,
    'NumHeteroatoms': Lipinski.NumHeteroatoms,
    'FractionCSP3': Lipinski.FractionCSP3,
    
    # Surface properties
    'TPSA': MolSurf.TPSA,
    'LabuteASA': MolSurf.LabuteASA,
    
    # Topological properties
    'BalabanJ': GraphDescriptors.BalabanJ,
    'BertzCT': GraphDescriptors.BertzCT,
    'Chi0v': GraphDescriptors.Chi0v,
    'Chi1v': GraphDescriptors.Chi1v,
    'Chi2v': GraphDescriptors.Chi2v,
    'Chi3v': GraphDescriptors.Chi3v,
    'Chi4v': GraphDescriptors.Chi4v,
    
    # Drug-likeness properties
    'QED': Descriptors.qed,
    'NumRings': Descriptors.RingCount,
    'HeavyAtomCount': Descriptors.HeavyAtomCount,
    'FormalCharge': lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
    'NHOHCount': Descriptors.NHOHCount,
    'NOCount': Descriptors.NOCount
}

for idx, row in df3.iterrows():
    try:
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        
        if mol is not None:
            # Calculate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            
            # Convert fingerprint to a list of 0s and 1s
            fp_bits = list(fp.ToBitString())
            fp_bits = [int(b) for b in fp_bits]
            
            # Create a dictionary with fingerprint bits and physicochemical descriptors
            record = {}
            
            # Add Morgan fingerprint bits
            for i, bit in enumerate(fp_bits):
                record[f'Morgan_{i}'] = bit
                
            # Add physicochemical descriptors
            for desc_name, desc_func in physchem_descriptors.items():
                try:
                    value = desc_func(mol)
                    record[desc_name] = value
                except:
                    record[desc_name] = np.nan
            
            fingerprint_data.append(record)
        else:
            print(f"Warning: Could not parse SMILES: {row['canonical_smiles']}")
            failed_mols += 1
    except Exception as e:
        print(f"Error processing molecule {row['molecule_chembl_id']}: {e}")
        failed_mols += 1

print(f"Successfully calculated descriptors for {len(fingerprint_data)} compounds")
print(f"Failed to process {failed_mols} compounds")

# Create descriptor DataFrame
df3_X = pd.DataFrame(fingerprint_data)

# Handle missing values
df3_X = df3_X.fillna(df3_X.mean())

# Print summary of descriptor types
morgan_count = sum(1 for col in df3_X.columns if col.startswith('Morgan_'))
physchem_count = sum(1 for col in df3_X.columns if not col.startswith('Morgan_'))
print(f"Generated {morgan_count} Morgan fingerprint bits and {physchem_count} physicochemical descriptors")
print(f"Total descriptors: {df3_X.shape[1]}")
print(f"Descriptor matrix shape: {df3_X.shape}")

# Get the pIC50 values as the target variable
df3_Y = df3['pIC50']
print(f"Target variable: pIC50 with {len(df3_Y)} values")

# Combine the X and Y data into a single DataFrame
dataset3 = pd.concat([df3_X, df3_Y], axis=1)
print(f"Final dataset has {dataset3.shape[0]} compounds and {dataset3.shape[1]} features (including pIC50)")

# Save the final dataset for model building
output_file = 'pancreatic_cancer_bioactivity_data_3class_pIC50_pubchem_fp.csv'
dataset3.to_csv(output_file, index=False)
print(f"Saved final dataset to {output_file}")

# List files created for the next step
created_files = [f for f in os.listdir() if os.path.isfile(f) and f.endswith('.csv')]
print("\nFiles generated for model building:")
for file in created_files:
    if file == output_file:
        print(f"- {file} ({os.path.getsize(file)/1024/1024:.2f} MB) <- Use this for modeling")
    elif file.endswith('.csv'):
        print(f"- {file} ({os.path.getsize(file)/1024/1024:.2f} MB)")

print("\nStep 3 completed. Run 4.Regression_Models_with_Random_Forest.py next.")


