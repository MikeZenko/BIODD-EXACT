#Importing libraries
import pandas as pd
import numpy as np
import os
import sys
import urllib.request
import urllib.parse
import time
import json
import ssl
from tqdm import tqdm

print("Step 1: Downloading Enhanced Pancreatic Cancer Bioactivity Data from ChEMBL")

# Disable SSL certificate verification (only for testing)
ssl_context = ssl._create_unverified_context()

# ChEMBL API endpoints
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_chembl_data(endpoint, params=None):
    """Get data from ChEMBL REST API with retry logic using urllib"""
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{CHEMBL_API_BASE}/{endpoint}?{query_string}"
    else:
        url = f"{CHEMBL_API_BASE}/{endpoint}"
        
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(req, context=ssl_context) as response:
                if response.getcode() == 200:
                    data = response.read().decode('utf-8')
                    return json.loads(data)
                else:
                    raise Exception(f"HTTP Error: {response.getcode()}")
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count} failed: {e}")
            if retry_count < max_retries:
                sleep_time = 3 * retry_count
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed to download data after multiple attempts")
                raise

def get_all_compounds(target_chembl_id, activity_type="IC50"):
    """Get all compounds for a target using pagination"""
    print(f"Retrieving all {activity_type} bioactivity data for target {target_chembl_id}...")
    
    all_activities = []
    limit = 100  # Number of records per page
    offset = 0
    total_count = None
    
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": activity_type,
        "standard_value__isnull": "false",
        "limit": limit,
        "offset": offset
    }
    
    # First request to get total count
    try:
        initial_data = get_chembl_data("activity", params)
        total_count = initial_data.get('page_meta', {}).get('total_count', 0)
        if 'activities' in initial_data:
            all_activities.extend(initial_data['activities'])
        
        print(f"Found {total_count} {activity_type} activity data points for {target_chembl_id}")
        
        # Fetch remaining pages
        if total_count > limit:
            pbar = tqdm(total=total_count, initial=len(all_activities), desc=f"Downloading {target_chembl_id} data")
            
            while len(all_activities) < total_count:
                offset += limit
                params["offset"] = offset
                page_data = get_chembl_data("activity", params)
                if 'activities' in page_data:
                    new_activities = page_data['activities']
                    all_activities.extend(new_activities)
                    pbar.update(len(new_activities))
                else:
                    break
            
            pbar.close()
    
    except Exception as e:
        print(f"Error fetching data for {target_chembl_id}: {e}")
    
    return all_activities

def get_target_info(target_chembl_id):
    """Get information about a target"""
    try:
        target_info = get_chembl_data(f"target/{target_chembl_id}")
        print(f"Target info retrieved: {target_info['target_type']} - {target_info['pref_name']}")
        return target_info
    except Exception as e:
        print(f"Error getting target info for {target_chembl_id}: {e}")
        return None

def fetch_smiles_for_molecules(molecule_ids):
    """Fetch SMILES for a list of molecule ChEMBL IDs"""
    print(f"Retrieving SMILES for {len(molecule_ids)} compounds...")
    smiles_dict = {}
    
    for mol_id in tqdm(molecule_ids, desc="Fetching SMILES"):
        try:
            mol_data = get_chembl_data(f"molecule/{mol_id}")
            if 'molecule_structures' in mol_data and mol_data['molecule_structures'] is not None:
                if 'canonical_smiles' in mol_data['molecule_structures']:
                    smiles_dict[mol_id] = mol_data['molecule_structures']['canonical_smiles']
        except Exception as e:
            print(f"Error getting SMILES for {mol_id}: {e}")
            continue
            
    return smiles_dict

def process_activities(activities, target_name):
    """Process activity data into a DataFrame with target information"""
    if not activities:
        return pd.DataFrame()
    
    df = pd.DataFrame(activities)
    
    # Add target name column
    df['target_name'] = target_name
    df['target_chembl_id'] = activities[0].get('target_chembl_id', '')
    
    # Check for required columns
    required_columns = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in data: {missing_columns}")
        
        # If SMILES is missing, try to get it
        if 'canonical_smiles' in missing_columns and 'molecule_chembl_id' in df.columns:
            # Get unique molecule IDs (limit to 500 to avoid long processing time)
            molecule_ids = df['molecule_chembl_id'].unique()[:500]
            smiles_dict = fetch_smiles_for_molecules(molecule_ids)
            
            df['canonical_smiles'] = df['molecule_chembl_id'].map(smiles_dict)
    
    return df

def preprocess_bioactivity_data(df):
    """Apply preprocessing steps to the bioactivity data"""
    if df.empty:
        return df
    
    print("Pre-processing bioactivity data...")
    
    # Keep only entries with the necessary data
    df2 = df.dropna(subset=['standard_value', 'canonical_smiles', 'molecule_chembl_id'])
    print(f"After removing entries with missing data: {len(df2)}")
    
    # Convert standard_value to numeric, handling any non-numeric values
    df2['standard_value'] = pd.to_numeric(df2['standard_value'], errors='coerce')
    df2 = df2.dropna(subset=['standard_value'])
    print(f"After converting to numeric values: {len(df2)}")
    
    # Classify compounds by activity
    bioactivity_class = []
    for i in df2['standard_value']:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")
    
    # Combine data
    selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'target_name', 'target_chembl_id']
    df3 = df2[selection]
    bioactivity_class = pd.Series(bioactivity_class, name='bioactivity_class')
    df4 = pd.concat([df3, bioactivity_class], axis=1)
    
    # Calculate pIC50
    def norm_value(input_df):
        norm = []
        for i in input_df['standard_value']:
            if i > 100000000:
                i = 100000000
            norm.append(i)
        input_df['standard_value_norm'] = norm
        return input_df
    
    def pIC50(input_df):
        pIC50_values = []
        for i in input_df['standard_value_norm']:
            molar = i*(10**-9)  # Converts nM to M
            pIC50_values.append(-np.log10(molar))
        input_df['pIC50'] = pIC50_values
        return input_df
    
    df5 = norm_value(df4.copy())
    df6 = pIC50(df5.copy())
    
    return df6

def save_target_information(targets_data):
    """Save target information to a CSV file"""
    targets_df = pd.DataFrame(targets_data)
    targets_df.to_csv('target_information.csv', index=False)
    print(f"Saved target information to target_information.csv")

try:
    # Define target ChEMBL IDs for pancreatic cancer
    targets = [
        {"id": "CHEMBL5619", "name": "PIM1 kinase"},
        {"id": "CHEMBL4523", "name": "PAK4"}
    ]
    
    # Store target information
    target_info_list = []
    
    # Get bioactivity data for all targets
    all_activities_df = pd.DataFrame()
    
    for target in targets:
        target_id = target["id"]
        target_name = target["name"]
        
        # Get target information
        target_info = get_target_info(target_id)
        if target_info:
            target_info_list.append({
                "target_chembl_id": target_id,
                "target_name": target_name,
                "target_type": target_info.get('target_type', ''),
                "organism": target_info.get('organism', ''),
                "pref_name": target_info.get('pref_name', '')
            })
        
        # Get activity data for this target
        activities = get_all_compounds(target_id)
        
        if activities:
            # Process activities into DataFrame
            target_df = process_activities(activities, target_name)
            
            # Append to the all activities dataframe
            all_activities_df = pd.concat([all_activities_df, target_df], ignore_index=True)
            
            print(f"Retrieved {len(target_df)} compounds for {target_name}")
        else:
            print(f"No activity data found for {target_name}")
    
    # Save raw combined data
    if not all_activities_df.empty:
        all_activities_df.to_csv('bioactivity_data_raw.csv', index=False)
        print(f"Saved raw data to bioactivity_data_raw.csv with {len(all_activities_df)} total compounds")
        
        # Save target information
        save_target_information(target_info_list)
        
        # Preprocess the data
        processed_df = preprocess_bioactivity_data(all_activities_df)
        
        # Save processed data
        processed_df.to_csv('bioactivity_data_preprocessed.csv', index=False)
        print(f"Saved preprocessed data to bioactivity_data_preprocessed.csv with {len(processed_df)} compounds")
        
        # Show file information
        print("\nFiles generated:")
        for file in ['bioactivity_data_raw.csv', 'bioactivity_data_preprocessed.csv', 'target_information.csv']:
            if os.path.exists(file):
                print(f"- {file} ({os.path.getsize(file)/1024:.1f} KB)")
    else:
        print("No activity data could be retrieved for any target")
        sys.exit(1)
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    print("Failed to download data from ChEMBL.")
    sys.exit(1)

print("\nStep 1 completed. Run 2.Exploratory_Data_Analysis.py next.")
