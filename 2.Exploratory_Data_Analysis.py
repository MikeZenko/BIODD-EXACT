import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

# Set plot style
plt.style.use('ggplot')
sns.set(style='whitegrid', font_scale=1.2)

print("Step 2: Performing Exploratory Data Analysis on Bioactivity Data")

# Check if data file exists
try:
    df = pd.read_csv('bioactivity_data_preprocessed.csv')
    print(f"Loaded preprocessed data with {len(df)} compounds")
except FileNotFoundError:
    print("Error: bioactivity_data_preprocessed.csv not found. Run Step 1 first.")
    exit()

# Check for any NaN or non-string values in SMILES
if df['canonical_smiles'].isna().any():
    print(f"WARNING: Found {df['canonical_smiles'].isna().sum()} missing SMILES values")
    df = df.dropna(subset=['canonical_smiles'])
    print(f"Dropped rows with missing SMILES. Remaining: {len(df)}")

# Ensure SMILES are strings
df['canonical_smiles'] = df['canonical_smiles'].astype(str)

print("\n1. Basic Data Overview")
print("-" * 40)
print(f"Shape of the dataset: {df.shape}")
print(f"Number of unique compounds: {df['molecule_chembl_id'].nunique()}")
print(f"Number of unique targets: {df['target_chembl_id'].nunique()}")

# Display target distribution
print("\nDistribution by target:")
target_counts = df['target_name'].value_counts()
for target, count in target_counts.items():
    print(f"- {target}: {count} compounds")

# Display class distribution
print("\nBioactivity class distribution:")
class_counts = df['bioactivity_class'].value_counts()
for cls, count in class_counts.items():
    print(f"- {cls}: {count} compounds ({count/len(df)*100:.1f}%)")

# Calculate Lipinski descriptors
def lipinski(smiles):
    """Calculate Lipinski descriptors for a molecule"""
    moldata = []
    for smile in smiles:
        try:
            # Convert to string if not already
            if not isinstance(smile, str):
                smile = str(smile)
                
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:  # Ensure valid SMILES
                moldata.append(
                    {
                        'MW': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'NumHDonors': Lipinski.NumHDonors(mol),
                        'NumHAcceptors': Lipinski.NumHAcceptors(mol)
                    }
                )
            else:
                print(f"Warning: Could not parse SMILES: {smile}")
                # Add placeholder values for invalid SMILES
                moldata.append(
                    {
                        'MW': 0,
                        'LogP': 0,
                        'NumHDonors': 0,
                        'NumHAcceptors': 0
                    }
                )
        except Exception as e:
            print(f"Error processing SMILES: {smile}. Error: {e}")
            # Add placeholder values for error cases
            moldata.append(
                {
                    'MW': 0,
                    'LogP': 0,
                    'NumHDonors': 0,
                    'NumHAcceptors': 0
                }
            )
    return pd.DataFrame(moldata)

# Calculate descriptors
print("\n2. Calculating Lipinski Descriptors")
print("-" * 40)
lipinski_df = lipinski(df['canonical_smiles'])

# Combine with original data
df2 = pd.concat([df, lipinski_df], axis=1)

# Create output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# Create subdirectories for target-specific plots
if not os.path.exists('plots/targets'):
    os.makedirs('plots/targets')

# Save processed data with descriptors
df2.to_csv('pancreatic_cancer_bioactivity_data_2class_lipinski.csv', index=False)
print(f"Saved data with Lipinski descriptors to pancreatic_cancer_bioactivity_data_2class_lipinski.csv")

# Convert to two-class for analysis (active vs. inactive)
df2_2class = df2[df2['bioactivity_class'] != 'intermediate']
print(f"\nExcluding intermediate compounds, remaining: {len(df2_2class)}")

# Separate active and inactive
active = df2_2class[df2_2class['bioactivity_class'] == 'active']
inactive = df2_2class[df2_2class['bioactivity_class'] == 'inactive']

print(f"Active compounds: {len(active)}")
print(f"Inactive compounds: {len(inactive)}")

# Statistical analysis
print("\n3. Statistical Analysis")
print("-" * 40)

def perform_ttest(descriptor):
    """Perform t-test between active and inactive compounds for a descriptor"""
    from scipy.stats import ttest_ind
    stat, pvalue = ttest_ind(active[descriptor], inactive[descriptor])
    return {
        'descriptor': descriptor,
        'active_mean': active[descriptor].mean(),
        'inactive_mean': inactive[descriptor].mean(),
        'difference': active[descriptor].mean() - inactive[descriptor].mean(),
        'p_value': pvalue,
        'significant': 'Yes' if pvalue < 0.05 else 'No'
    }

# Perform t-tests for all Lipinski descriptors
descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'pIC50']
results = []
for desc in descriptors:
    results.append(perform_ttest(desc))

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.round(3)
print("\nDifferences between active and inactive compounds:")
print(results_df.to_string(index=False))

# Plot distributions by target and activity
print("\n4. Visualization")
print("-" * 40)

# Target distribution pie chart
plt.figure(figsize=(8, 6))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Compounds by Target')
plt.tight_layout()
plt.savefig('plots/target_distribution.png', dpi=300)
print("Saved target distribution plot to plots/target_distribution.png")

# Bioactivity class distribution by target
plt.figure(figsize=(10, 6))
target_class = pd.crosstab(df['target_name'], df['bioactivity_class'])
target_class.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Bioactivity Class Distribution by Target')
plt.xlabel('Target')
plt.ylabel('Number of Compounds')
plt.tight_layout()
plt.savefig('plots/bioactivity_by_target.png', dpi=300)
print("Saved bioactivity class distribution by target plot to plots/bioactivity_by_target.png")

# pIC50 value distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='pIC50', hue='target_name', kde=True, bins=30, element='step')
plt.title('Distribution of pIC50 Values by Target')
plt.xlabel('pIC50')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/pIC50_distribution_by_target.png', dpi=300)
print("Saved pIC50 distribution by target plot to plots/pIC50_distribution_by_target.png")

# Scatter plot: MW vs. LogP colored by activity class
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df2_2class, x='MW', y='LogP', hue='bioactivity_class', alpha=0.7)
plt.title('Molecular Weight vs. LogP')
plt.xlabel('Molecular Weight')
plt.ylabel('LogP')
plt.axhline(y=5, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold for LogP
plt.axvline(x=500, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold for MW
plt.legend(title='Bioactivity')
plt.tight_layout()
plt.savefig('plots/mw_vs_logp.png', dpi=300)
print("Saved MW vs. LogP plot to plots/mw_vs_logp.png")

# Boxplots for Lipinski descriptors
plt.figure(figsize=(15, 10))

# MW
plt.subplot(2, 2, 1)
sns.boxplot(x='bioactivity_class', y='MW', data=df2_2class)
plt.title('Molecular Weight')
plt.axhline(y=500, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold

# LogP
plt.subplot(2, 2, 2)
sns.boxplot(x='bioactivity_class', y='LogP', data=df2_2class)
plt.title('LogP')
plt.axhline(y=5, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold

# NumHDonors
plt.subplot(2, 2, 3)
sns.boxplot(x='bioactivity_class', y='NumHDonors', data=df2_2class)
plt.title('Number of H-Bond Donors')
plt.axhline(y=5, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold

# NumHAcceptors
plt.subplot(2, 2, 4)
sns.boxplot(x='bioactivity_class', y='NumHAcceptors', data=df2_2class)
plt.title('Number of H-Bond Acceptors')
plt.axhline(y=10, color='red', linestyle='--', alpha=0.5)  # Lipinski threshold

plt.tight_layout()
plt.savefig('plots/lipinski_boxplots.png', dpi=300)
print("Saved Lipinski descriptor boxplots to plots/lipinski_boxplots.png")

# Compare pIC50 distribution between targets
plt.figure(figsize=(10, 6))
sns.boxplot(x='target_name', y='pIC50', data=df)
plt.title('pIC50 Distribution by Target')
plt.xlabel('Target')
plt.ylabel('pIC50')
plt.tight_layout()
plt.savefig('plots/pIC50_by_target.png', dpi=300)
print("Saved pIC50 by target plot to plots/pIC50_by_target.png")

# Generate separate plots for each target
for target in df['target_name'].unique():
    # Get target-specific data from df2 which has the lipinski descriptors
    target_df = df2[df2['target_name'] == target]
    target_df_2class = target_df[target_df['bioactivity_class'] != 'intermediate']
    
    # Skip if too few compounds
    if len(target_df_2class) < 10:
        continue
    
    # pIC50 distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=target_df, x='pIC50', hue='bioactivity_class', kde=True, bins=20)
    plt.title(f'Distribution of pIC50 Values for {target}')
    plt.xlabel('pIC50')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'plots/targets/{target.replace(" ", "_")}_pIC50_distribution.png', dpi=300)
    
    # Lipinski violations - calculate here directly
    target_df['lipinski_violations'] = ((target_df['MW'] > 500).astype(int) + 
                                    (target_df['LogP'] > 5).astype(int) + 
                                    (target_df['NumHDonors'] > 5).astype(int) + 
                                    (target_df['NumHAcceptors'] > 10).astype(int))
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=target_df, x='lipinski_violations', hue='bioactivity_class')
    plt.title(f'Lipinski Violations for {target}')
    plt.xlabel('Number of Violations')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'plots/targets/{target.replace(" ", "_")}_lipinski_violations.png', dpi=300)

# Lipinski Rule of 5 violations
df2['lipinski_violations'] = ((df2['MW'] > 500).astype(int) + 
                             (df2['LogP'] > 5).astype(int) + 
                             (df2['NumHDonors'] > 5).astype(int) + 
                             (df2['NumHAcceptors'] > 10).astype(int))

plt.figure(figsize=(10, 6))
sns.countplot(data=df2, x='lipinski_violations', hue='bioactivity_class')
plt.title('Number of Lipinski Rule Violations')
plt.xlabel('Number of Violations')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/lipinski_violations.png', dpi=300)
print("Saved Lipinski violations plot to plots/lipinski_violations.png")

# Correlation matrix of descriptors
plt.figure(figsize=(10, 8))
correlation_matrix = df2[['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'pIC50']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Molecular Descriptors')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300)
print("Saved correlation matrix to plots/correlation_matrix.png")

# Mann-Whitney U test for pIC50 between targets
from scipy.stats import mannwhitneyu

if df['target_chembl_id'].nunique() > 1:
    targets = df['target_chembl_id'].unique()
    
    if len(targets) == 2:
        group1 = df[df['target_chembl_id'] == targets[0]]['pIC50']
        group2 = df[df['target_chembl_id'] == targets[1]]['pIC50']
        
        u_stat, p_value = mannwhitneyu(group1, group2)
        
        print(f"\nMann-Whitney U test for pIC50 between {targets[0]} and {targets[1]}:")
        print(f"U statistic: {u_stat}")
        print(f"p-value: {p_value}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Create a boxplot comparing the two targets
        plt.figure(figsize=(8, 6))
        target_map = {targets[0]: df[df['target_chembl_id'] == targets[0]]['target_name'].iloc[0],
                      targets[1]: df[df['target_chembl_id'] == targets[1]]['target_name'].iloc[0]}
        
        # Create comparison dataframe with reset_index to avoid duplicate indices
        group1_df = pd.DataFrame({'pIC50': group1, 'Target': target_map[targets[0]]}).reset_index(drop=True)
        group2_df = pd.DataFrame({'pIC50': group2, 'Target': target_map[targets[1]]}).reset_index(drop=True)
        comparison_df = pd.concat([group1_df, group2_df], ignore_index=True)
        
        sns.boxplot(x='Target', y='pIC50', data=comparison_df)
        plt.title('Comparison of pIC50 Values Between Targets')
        plt.annotate(f'p-value: {p_value:.5f}', xy=(0.5, 0.05), xycoords='figure fraction', 
                    ha='center', fontsize=12, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('plots/target_comparison_pIC50.png', dpi=300)
        print("Saved target comparison plot to plots/target_comparison_pIC50.png")

print("\nStep 2 completed. Run 3.Descriptor_Calculation_and_Dataset_Preparation.py next.")