#!/usr/bin/env python3
"""
Script to collect trial-level NLL data from all evaluation sources:
- Zero-shot evaluations
- Cognitive model baselines  
- Original Centaur results
- Context-free evaluations

Output: Consolidated dataset for statistical testing
"""

import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pickle

def load_pth_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load .pth file safely."""
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e2:
            print(f"Failed to load {file_path}: {e}, {e2}")
            return None

def extract_zero_shot_nlls(dataset_name: str) -> Optional[np.ndarray]:
    """Extract trial-level NLLs from zero-shot evaluation results."""
    
    # Try different file types in order of preference
    file_patterns = [
        f"eval_results/{dataset_name}_detailed_evaluation_results.json",
        f"eval_results/{dataset_name}_comprehensive_zero_shot_results.json"
    ]
    
    for file_path in file_patterns:
        if not Path(file_path).exists():
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            nlls = []
            
            # Handle detailed_evaluation_results.json structure
            if 'per_trial_results' in data and isinstance(data['per_trial_results'], list):
                for result in data['per_trial_results']:
                    if 'trial_nll' in result:
                        nlls.append(result['trial_nll'])
                    elif 'nll' in result:
                        nlls.append(result['nll'])
            
            # Handle comprehensive_zero_shot_results.json structure  
            elif ('evaluation_results' in data and 
                  isinstance(data['evaluation_results'], dict)):
                # Look for detailed results with trial-level data
                for eval_type, eval_data in data['evaluation_results'].items():
                    if (isinstance(eval_data, dict) and 
                        'detailed_results' in eval_data and
                        'per_trial_results' in eval_data['detailed_results']):
                        for trial in eval_data['detailed_results']['per_trial_results']:
                            if 'trial_nll' in trial:
                                nlls.append(trial['trial_nll'])
                            elif 'nll' in trial:
                                nlls.append(trial['nll'])
            
            # Handle other possible structures
            elif 'results' in data and isinstance(data['results'], list):
                for result in data['results']:
                    if 'trial_nll' in result:
                        nlls.append(result['trial_nll'])
                    elif 'nll' in result:
                        nlls.append(result['nll'])
            
            if nlls:
                return np.array(nlls)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return None

def extract_context_free_nlls(dataset_name: str) -> Optional[np.ndarray]:
    """Extract trial-level NLLs from context-free evaluation results."""
    file_path = f"context_free_eval/{dataset_name}_context_free_results.json"
    
    print(f"    Looking for context-free file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"    Context-free file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"    Context-free file loaded successfully")
        
        nlls = []
        
        # Handle context-free evaluation structure
        if ('evaluation_results' in data and 
            'context_free_centaur' in data['evaluation_results'] and
            'detailed_results' in data['evaluation_results']['context_free_centaur'] and
            'per_trial_results' in data['evaluation_results']['context_free_centaur']['detailed_results']):
            
            trials = data['evaluation_results']['context_free_centaur']['detailed_results']['per_trial_results']
            print(f"    Found {len(trials)} trials in context-free results")
            
            for trial in trials:
                if 'trial_nll' in trial and trial['trial_nll'] != float('inf'):
                    nlls.append(trial['trial_nll'])
                elif 'nll' in trial and trial['nll'] != float('inf'):
                    nlls.append(trial['nll'])
        
        # Alternative structure (just in case)
        elif 'per_trial_results' in data:
            trials = data['per_trial_results']
            print(f"    Found {len(trials)} trials in alternative structure")
            
            for trial in trials:
                if 'trial_nll' in trial and trial['trial_nll'] != float('inf'):
                    nlls.append(trial['trial_nll'])
                elif 'nll' in trial and trial['nll'] != float('inf'):
                    nlls.append(trial['nll'])
        
        if nlls:
            print(f"    Extracted {len(nlls)} valid NLLs")
            return np.array(nlls)
        else:
            print(f"    No valid NLLs found in context-free results")
            return None
        
    except Exception as e:
        print(f"    Error extracting context-free NLLs for {dataset_name}: {e}")
        return None

def extract_baseline_nlls(dataset_name: str, baseline_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract trial-level NLLs from cognitive model baseline data."""
    if not isinstance(baseline_data, dict):
        return None
    
    # Direct match
    if dataset_name in baseline_data:
        data = baseline_data[dataset_name]
        if isinstance(data, (list, np.ndarray, torch.Tensor)):
            return np.array(data)
    
    # Special case for collsioo2023MCPL
    if dataset_name == "collsioo2023MCPL_all":
        special_key = "collsiöö2023MCPL"
        if special_key in baseline_data:
            data = baseline_data[special_key]
            if isinstance(data, (list, np.ndarray, torch.Tensor)):
                return np.array(data)
    
    # Fuzzy matching
    for key in baseline_data.keys():
        if (key.startswith(dataset_name) or dataset_name.startswith(key) or 
            dataset_name in key or key in dataset_name):
            data = baseline_data[key]
            if isinstance(data, (list, np.ndarray, torch.Tensor)):
                return np.array(data)
    
    return None

def extract_centaur_nlls(dataset_name: str, centaur_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract trial-level NLLs from original Centaur data."""
    if not isinstance(centaur_data, dict):
        return None
    
    # Direct match
    if dataset_name in centaur_data:
        data = centaur_data[dataset_name]
        if isinstance(data, (list, np.ndarray, torch.Tensor)):
            return np.array(data)
    
    # Special case for collsioo2023MCPL
    if dataset_name == "collsioo2023MCPL_all":
        special_key = "collsiöö2023MCPL"
        if special_key in centaur_data:
            data = centaur_data[special_key]
            if isinstance(data, (list, np.ndarray, torch.Tensor)):
                return np.array(data)
    
    # Fuzzy matching
    for key in centaur_data.keys():
        if (key.startswith(dataset_name) or dataset_name.startswith(key) or 
            dataset_name in key or key in dataset_name):
            data = centaur_data[key]
            if isinstance(data, (list, np.ndarray, torch.Tensor)):
                return np.array(data)
    
    return None

def get_all_evaluated_datasets() -> List[str]:
    """Get all datasets we have evaluated across all methods."""
    datasets = set()
    
    # From zero-shot evaluations
    eval_results_dir = Path("eval_results")
    if eval_results_dir.exists():
        for file in eval_results_dir.glob("*_detailed_evaluation_results.json"):
            dataset_name = file.stem.replace("_detailed_evaluation_results", "")
            datasets.add(dataset_name)
        for file in eval_results_dir.glob("*_comprehensive_zero_shot_results.json"):
            dataset_name = file.stem.replace("_comprehensive_zero_shot_results", "")
            datasets.add(dataset_name)
        for file in eval_results_dir.glob("*_basic_evaluation_results.json"):
            dataset_name = file.stem.replace("_basic_evaluation_results", "")
            datasets.add(dataset_name)
    
    # From context-free evaluations
    context_free_dir = Path("context_free_eval")
    if context_free_dir.exists():
        for file in context_free_dir.glob("*_context_free_results.json"):
            dataset_name = file.stem.replace("_context_free_results", "")
            datasets.add(dataset_name)
            print(f"  Found context-free results for: {dataset_name}")
    
    # Add known datasets that might have different naming
    known_datasets = [
        "ruggeri2022globalizability",
        "hilbig2014generalized", 
        "hebart2023things",
        "collsioo2023MCPL_all",
        "wu2018generalisation_exp1",
        "dubois2022value"
    ]
    
    for dataset in known_datasets:
        datasets.add(dataset)
    
    return sorted(list(datasets))

def get_random_nll_for_dataset(dataset_name: str) -> float:
    """Get the theoretical random guessing NLL for each dataset."""
    random_nlls = {
        "ruggeri2022globalizability": 0.6931471805599453,  # ln(2) for binary choice
        "hilbig2014generalized": 0.6931471805599453,       # ln(2) for binary choice  
        "hebart2023things": 1.0986122886681098,            # ln(3) for 3-choice
        "collsioo2023MCPL_all": 2.1972245773362196,        # ln(9) for 9-choice (Caldionine 1-9)
        "wu2018generalisation_exp1": 3.4011973816621555,   # ln(30) for 30-choice
        "dubois2022value": 0.6931471805599453,             # ln(2) for binary choice
    }
    return random_nlls.get(dataset_name, 0.6931471805599453)  # Default to binary choice

def generate_random_baseline_nlls(trial_count: int, dataset_name: str) -> np.ndarray:
    """Generate random guessing NLL array for statistical comparison."""
    if trial_count <= 0:
        return None
    
    random_nll = get_random_nll_for_dataset(dataset_name)
    # For perfect random guessing, all trials would have the same NLL
    # But we can add tiny noise to simulate sampling variation
    random_nlls = np.full(trial_count, random_nll)
    
    # Add minimal noise to simulate realistic variation (±0.1% of the value)
    noise_std = random_nll * 0.001
    noise = np.random.normal(0, noise_std, trial_count)
    random_nlls += noise
    
    return random_nlls

def collect_all_trial_nlls():
    """Collect trial-level NLL data from all sources."""
    print("="*80)
    print("COLLECTING TRIAL-LEVEL NLL DATA")
    print("="*80)
    
    # Load original data
    baseline_path = "original/results/custom_metrics_full_log_likelihoods_baselines.pth"
    centaur_path = "original/results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth"
    
    baseline_data = load_pth_file(baseline_path) if Path(baseline_path).exists() else None
    centaur_data = load_pth_file(centaur_path) if Path(centaur_path).exists() else None
    
    print(f"Baseline data loaded: {baseline_data is not None}")
    print(f"Centaur data loaded: {centaur_data is not None}")
    
    # Get all datasets
    datasets = get_all_evaluated_datasets()
    print(f"\nFound {len(datasets)} datasets to process:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i:2d}. {dataset}")
    
    # Collect data for each dataset
    all_data = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_data = {
            "dataset_name": dataset_name,
            "zero_shot_nlls": None,
            "context_free_nlls": None,
            "baseline_nlls": None,
            "original_centaur_nlls": None,
            "random_baseline_nlls": None,
            "counts": {}
        }
        
        # Extract zero-shot NLLs
        zero_shot_nlls = extract_zero_shot_nlls(dataset_name)
        if zero_shot_nlls is not None:
            dataset_data["zero_shot_nlls"] = zero_shot_nlls
            dataset_data["counts"]["zero_shot"] = len(zero_shot_nlls)
            print(f"  Zero-shot: {len(zero_shot_nlls)} trials")
        else:
            dataset_data["counts"]["zero_shot"] = 0
            print(f"  Zero-shot: No data")
        
        # Extract context-free NLLs
        context_free_nlls = extract_context_free_nlls(dataset_name)
        if context_free_nlls is not None:
            dataset_data["context_free_nlls"] = context_free_nlls
            dataset_data["counts"]["context_free"] = len(context_free_nlls)
            print(f"  Context-free: {len(context_free_nlls)} trials")
        else:
            dataset_data["counts"]["context_free"] = 0
            print(f"  Context-free: No data")
        
        # Extract baseline NLLs
        if baseline_data:
            baseline_nlls = extract_baseline_nlls(dataset_name, baseline_data)
            if baseline_nlls is not None:
                dataset_data["baseline_nlls"] = baseline_nlls
                dataset_data["counts"]["baseline"] = len(baseline_nlls)
                print(f"  Baseline: {len(baseline_nlls)} trials")
            else:
                dataset_data["counts"]["baseline"] = 0
                print(f"  Baseline: No data")
        else:
            dataset_data["counts"]["baseline"] = 0
            print(f"  Baseline: No data (file not loaded)")
        
        # Extract original Centaur NLLs
        if centaur_data:
            centaur_nlls = extract_centaur_nlls(dataset_name, centaur_data)
            if centaur_nlls is not None:
                dataset_data["original_centaur_nlls"] = centaur_nlls
                dataset_data["counts"]["original_centaur"] = len(centaur_nlls)
                print(f"  Original Centaur: {len(centaur_nlls)} trials")
            else:
                dataset_data["counts"]["original_centaur"] = 0
                print(f"  Original Centaur: No data")
        else:
            dataset_data["counts"]["original_centaur"] = 0
            print(f"  Original Centaur: No data (file not loaded)")
        
        # Generate random baseline for statistical comparison
        # Use the maximum trial count available from any method
        max_trials = max([count for count in dataset_data["counts"].values() if count > 0], default=0)
        if max_trials > 0:
            random_nlls = generate_random_baseline_nlls(max_trials, dataset_name)
            dataset_data["random_baseline_nlls"] = random_nlls
            dataset_data["counts"]["random_baseline"] = len(random_nlls)
            theoretical_nll = get_random_nll_for_dataset(dataset_name)
            print(f"  Random baseline: {len(random_nlls)} trials (theoretical NLL: {theoretical_nll:.4f})")
        else:
            dataset_data["counts"]["random_baseline"] = 0
            print(f"  Random baseline: No data (no reference trial count)")
        
        # Check consistency
        counts = [count for count in dataset_data["counts"].values() if count > 0]
        if len(set(counts)) <= 1:
            print(f"  ✓ CONSISTENT: All non-empty datasets have same trial count")
        else:
            print(f"  ✗ INCONSISTENT: Trial counts vary: {dataset_data['counts']}")
        
        all_data[dataset_name] = dataset_data
    
    # Create summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    eval_types = ["zero_shot", "context_free", "baseline", "original_centaur", "random_baseline"]
    summary_stats = {eval_type: {"datasets": 0, "total_trials": 0} for eval_type in eval_types}
    
    for dataset_name, data in all_data.items():
        for eval_type in eval_types:
            if data["counts"][eval_type] > 0:
                summary_stats[eval_type]["datasets"] += 1
                summary_stats[eval_type]["total_trials"] += data["counts"][eval_type]
    
    for eval_type, stats in summary_stats.items():
        print(f"{eval_type.replace('_', ' ').title()}: {stats['datasets']} datasets, {stats['total_trials']} total trials")
    
    # Save data in multiple formats for statistical analysis
    save_consolidated_data(all_data)
    
    return all_data

def save_consolidated_data(all_data: Dict[str, Any]):
    """Save consolidated data in multiple formats for statistical analysis."""
    
    # 1. Save as compressed numpy archive
    arrays_to_save = {}
    for dataset_name, data in all_data.items():
        for eval_type in ["zero_shot", "context_free", "baseline", "original_centaur", "random_baseline"]:
            nlls = data[f"{eval_type}_nlls"]
            if nlls is not None:
                key = f"{dataset_name}_{eval_type}_nlls"
                arrays_to_save[key] = nlls
    
    if arrays_to_save:
        np.savez_compressed("trial_level_nlls_all_methods.npz", **arrays_to_save)
        print(f"\n✓ Saved numpy arrays to: trial_level_nlls_all_methods.npz")
    
    # 2. Save metadata as JSON
    metadata = {}
    for dataset_name, data in all_data.items():
        metadata[dataset_name] = {
            "counts": data["counts"],
            "has_data": {
                eval_type: data[f"{eval_type}_nlls"] is not None 
                for eval_type in ["zero_shot", "context_free", "baseline", "original_centaur", "random_baseline"]
            },
            "theoretical_random_nll": get_random_nll_for_dataset(dataset_name)
        }
    
    with open("trial_level_nlls_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: trial_level_nlls_metadata.json")
    
    # 3. Create long-format DataFrame for statistical analysis
    long_format_data = []
    
    for dataset_name, data in all_data.items():
        for eval_type in ["zero_shot", "context_free", "baseline", "original_centaur", "random_baseline"]:
            nlls = data[f"{eval_type}_nlls"]
            if nlls is not None:
                for i, nll in enumerate(nlls):
                    long_format_data.append({
                        "dataset": dataset_name,
                        "evaluation_type": eval_type,
                        "trial_index": i,
                        "nll": float(nll)
                    })
    
    if long_format_data:
        df = pd.DataFrame(long_format_data)
        df.to_csv("trial_level_nlls_long_format.csv", index=False)
        print(f"✓ Saved long-format CSV to: trial_level_nlls_long_format.csv")
        
        # Create summary by dataset and evaluation type
        summary_df = df.groupby(["dataset", "evaluation_type"]).agg({
            "nll": ["count", "mean", "std", "min", "max", "median"]
        }).round(4)
        summary_df.columns = ["_".join(col).strip() for col in summary_df.columns]
        summary_df.to_csv("trial_level_nlls_summary.csv")
        print(f"✓ Saved summary CSV to: trial_level_nlls_summary.csv")
    
    print(f"\nData collection complete! Files saved:")
    print(f"  - trial_level_nlls_all_methods.npz (numpy arrays)")
    print(f"  - trial_level_nlls_metadata.json (metadata)")
    print(f"  - trial_level_nlls_long_format.csv (for statistical analysis)")
    print(f"  - trial_level_nlls_summary.csv (summary statistics)")
    print(f"\nRandom baselines included for statistical testing!")
    print(f"Each dataset now has random guessing NLLs matching the trial count of actual data.")

if __name__ == "__main__":
    print("Collecting trial-level NLL data from all evaluation methods")
    collect_all_trial_nlls() 