#!/usr/bin/env python3
"""
Visualize All Datasets Script
Automatically generates comprehensive visualizations for all available datasets.
Creates one figure per task, combining zero-shot and history-only results with baselines.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

# Try importing scienceplots for original paper styling
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("⚠️ scienceplots not available - install with: pip install scienceplots")

def extract_nll_from_results(results_file):
    """Extract NLL value from results file"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Try different locations for NLL value
        if isinstance(data, dict):
            # Basic results format
            if 'nll' in data:
                return data['nll'], data.get('num_samples', 0)
            
            # Comprehensive results format
            if 'evaluation_results' in data:
                eval_results = data['evaluation_results']
                
                # Zero-shot results
                if 'zero_shot_centaur' in eval_results:
                    nll = eval_results['zero_shot_centaur'].get('nll')
                    samples = 0
                    if 'detailed_results' in eval_results['zero_shot_centaur']:
                        detailed = eval_results['zero_shot_centaur']['detailed_results']
                        if 'summary_statistics' in detailed:
                            samples = detailed['summary_statistics'].get('valid_trials', 0)
                    return nll, samples
                
                # History-only results
                if 'context_free_centaur' in eval_results:
                    nll = eval_results['context_free_centaur'].get('nll')
                    samples = 0
                    if 'detailed_results' in eval_results['context_free_centaur']:
                        detailed = eval_results['context_free_centaur']['detailed_results']
                        if 'summary_statistics' in detailed:
                            samples = detailed['summary_statistics'].get('valid_trials', 0)
                    return nll, samples
        
        return None, 0
    except Exception as e:
        print(f"   Error reading {results_file}: {e}")
        return None, 0

def get_dataset_baselines(dataset_name):
    """Get baseline values for a dataset"""
    baselines = {
        "ruggeri2022globalizability": {
            "original_centaur": 0.4382756948471069,
            "cognitive_models": 0.6590430736541748,
            "random": 0.6931471805599453,
            "description": "Temporal decision making task"
        },
        "hilbig2014generalized": {
            "original_centaur": 0.0618637911975383,
            "cognitive_models": 0.1922362762758062,
            "random": 0.6931471805599453,
            "description": "Multi-attribute decision making task"
        },
        "wu2018generalisation_exp1": {
            "original_centaur": 1.8321380615234373,
            "cognitive_models": 2.792812824249268,
            "random": 3.4011973816621555,
            "description": "Multi-environment exploration task"
        },
        "collsioo2023MCPL_all": {
            "original_centaur": 1.123242735862732,
            "cognitive_models": 1.9236862861598658,
            "random": 2.1972245773362196,
            "description": "MCPL progressive learning task"
        },
        "hebart2023things": {
            "original_centaur": 0.7851982712745667,
            "cognitive_models": 0.8343300819396973,
            "random": 1.0986122886681098,
            "description": "THINGS odd-one-out task"
        }
    }
    
    return baselines.get(dataset_name, {})

def get_dataset_title(dataset_name):
    """Get appropriate title for each dataset"""
    titles = {
        "ruggeri2022globalizability": "Inter-temporal Choice (Non-sequential)\nRuggeri et al., 2022",
        "hilbig2014generalized": "Multi-attributes decision-making (Non-sequential)\nHilbig et al., 2014",
        "wu2018generalisation_exp1": "Spatially correlated multi-armed bandit task (Sequential)\nWu et al., 2018", 
        "collsioo2023MCPL_all": "Multi-cue Judgement (Sequential)\nCollsiöö et al., 2023",
        "hebart2023things": "THINGS odd-one-out (Non-sequential)\nHebart et al., 2023"
    }
    
    return titles.get(dataset_name, f"Dataset: {dataset_name.replace('_', ' ').title()}")

def create_comprehensive_plot(dataset_name, zero_shot_nll=None, history_only_nll=None, 
                             output_dir="all_datasets_plots"):
    """Create comprehensive comparison plot for a dataset"""
    print(f"📊 Creating plot for {dataset_name}")
    
    # Get dataset baselines
    baselines = get_dataset_baselines(dataset_name)
    if not baselines:
        print(f"   ❌ No baselines available for {dataset_name}")
        return None
    
    # Map datasets to specific cognitive model names
    cognitive_model_names = {
        "ruggeri2022globalizability": "Cognitive Models\n(Hyperbolic discounting)",
        "collsioo2023MCPL_all": "Cognitive Models\n(Linear regression)",
        "hilbig2014generalized": "Cognitive Models\n(Weighted additive)",
        "hebart2023things": "Cognitive Models\n(Odd-one-out)",
        "wu2018generalisation_exp1": "Cognitive Models\n(GP-UCB)"
    }
    
    cognitive_label = cognitive_model_names.get(dataset_name, "Cognitive\nModels")
    
    # Build comparison data with fixed ordering: Centaur, Cog Model, Context-free, Zero-shot
    comparison_data = []
    
    # 1. Original Centaur (always first)
    comparison_data.append({
        'Model': 'Original\nCentaur', 
        'NLL': baselines['original_centaur'], 
        'Type': 'Published Paper',
        'Method': 'Original'
    })
    
    # 2. Cognitive Models (always second) - with specific model name
    comparison_data.append({
        'Model': cognitive_label, 
        'NLL': baselines['cognitive_models'], 
        'Type': 'Published Paper',
        'Method': 'Cognitive'
    })
    
    # 3. Context-free (if available)
    if history_only_nll is not None:
        comparison_data.append({
            'Model': 'Centaur without\nPsychological Task', 
            'NLL': history_only_nll, 
            'Type': 'Our Evaluation',
            'Method': 'History-Only'
        })
    
    # 4. Zero-shot (if available)
    if zero_shot_nll is not None:
        comparison_data.append({
            'Model': 'Zero-Shot\nCentaur', 
            'NLL': zero_shot_nll, 
            'Type': 'Our Evaluation',
            'Method': 'Zero-Shot'
        })
    
    # Use original paper's styling if available
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['nature'])
    else:
        plt.style.use('default')
    
    # Create square plot - slightly wider to accommodate longer labels
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.5))
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Color scheme for different evaluation methods
    colors = []
    for _, row in comp_df.iterrows():
        if row['Method'] == 'Original':
            colors.append('#69005f')  # Purple for original Centaur
        elif row['Method'] == 'Cognitive':
            colors.append('#cbc9e2')  # Light purple for cognitive models
        elif row['Method'] == 'History-Only':
            colors.append('#ff7f0e')  # Orange for history-only
        elif row['Method'] == 'Zero-Shot':
            colors.append('#1f77b4')  # Blue for zero-shot
        else:
            colors.append('#2ca02c')  # Green for others
    
    # Create bars with appropriate width for square format
    bars = ax.bar(range(len(comp_df)), comp_df['NLL'], color=colors, alpha=0.8, width=0.65)
    
    # Styling adjusted for square format with longer labels
    ax.set_ylabel('Negative log-likelihood', fontsize=10.5)
    
    # Set dataset-specific title with adjusted padding for square format
    title = get_dataset_title(dataset_name)
    ax.set_title(title, fontsize=10.5, pad=8)
    
    ax.set_xticks(range(len(comp_df)))
    ax.set_xticklabels(comp_df['Model'], rotation=0, ha='center', fontsize=6.5)
    
    # Smart Y-axis handling for random baseline
    random_nll = baselines['random']
    model_nlls = comp_df['NLL'].tolist()
    max_model_nll = max(model_nlls)
    min_model_nll = min(model_nlls)
    model_range = max_model_nll - min_model_nll
    
    # Set intelligent y-limits
    y_bottom = max(0, min_model_nll - 0.1 * model_range)
    y_top = max(random_nll + 0.05 * random_nll, max_model_nll + 0.15 * model_range)
    ax.set_ylim(y_bottom, y_top)
    
    # Add random baseline line
    ax.axhline(y=random_nll, color='gray', linestyle='--', linewidth=1.8, alpha=0.7)
    ax.text(len(comp_df)/2 - 1.0, random_nll + (y_top - y_bottom) * 0.03, 'Random guessing', 
           fontsize=8, color='gray', horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    
    # Smart Y-axis tick spacing if random is much higher
    random_gap = random_nll - max_model_nll
    if random_gap > 0.5:
        # Dense ticks in model comparison range
        model_ticks = np.linspace(min_model_nll, max_model_nll, 4)
        # Add random baseline and one intermediate
        mid_point = (max_model_nll + random_nll) / 2
        random_ticks = [mid_point, random_nll]
        all_ticks = np.concatenate([model_ticks, random_ticks])
        all_ticks = np.unique(np.round(all_ticks, 3))
        ax.set_yticks(all_ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in all_ticks], fontsize=8)
        
        # Add subtle grid lines in the gap region
        for tick in all_ticks:
            if tick > max_model_nll + 0.1:
                ax.axhline(y=tick, color='lightgray', linewidth=0.5, alpha=0.2, zorder=0)
    else:
        # Set tick label font size for normal cases too
        ax.tick_params(axis='y', labelsize=8)
    
    # Add value labels on bars with adjusted positioning for square format
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_model_nll - min_model_nll) * 0.025,
               f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust tick label size for x-axis with tighter spacing
    ax.tick_params(axis='x', labelsize=6.5, pad=1)
    
    plt.tight_layout()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save plots
    jpg_file = output_path / f'{dataset_name}_comprehensive_comparison.jpg'
    pdf_file = output_path / f'{dataset_name}_comprehensive_comparison.pdf'
    png_file = output_path / f'{dataset_name}_comprehensive_comparison.png'
    
    plt.savefig(jpg_file, dpi=300, bbox_inches='tight', format='jpeg')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Plot saved: {jpg_file}")
    
    return comparison_data

def find_available_datasets():
    """Find all datasets with available results"""
    datasets = []
    
    # Check eval_results directory for zero-shot results
    eval_dir = Path("eval_results")
    if eval_dir.exists():
        for file in eval_dir.glob("*_basic_evaluation_results.json"):
            dataset = file.stem.replace("_basic_evaluation_results", "")
            if dataset not in datasets:
                datasets.append(dataset)
        
        for file in eval_dir.glob("*_comprehensive_zero_shot_results.json"):
            dataset = file.stem.replace("_comprehensive_zero_shot_results", "")
            if dataset not in datasets:
                datasets.append(dataset)
    
    # Check context_free_eval directory for history-only results
    context_dir = Path("context_free_eval")
    if context_dir.exists():
        for file in context_dir.glob("*_context_free_results.json"):
            dataset = file.stem.replace("_context_free_results", "")
            if dataset not in datasets:
                datasets.append(dataset)
    
    # Also check for datasets that have baselines even if no eval results
    baseline_datasets = ["ruggeri2022globalizability", "hilbig2014generalized", "wu2018generalisation_exp1", 
                        "collsioo2023MCPL_all", "hebart2023things"]
    for dataset in baseline_datasets:
        if dataset not in datasets:
            datasets.append(dataset)
    
    return sorted(datasets)

def process_dataset(dataset_name):
    """Process a single dataset and create visualization"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*60}")
    
    # Check for zero-shot results
    zero_shot_basic = f"eval_results/{dataset_name}_basic_evaluation_results.json"
    zero_shot_comprehensive = f"eval_results/{dataset_name}_comprehensive_zero_shot_results.json"
    
    zero_shot_nll = None
    zero_shot_samples = 0
    
    if Path(zero_shot_basic).exists():
        zero_shot_nll, zero_shot_samples = extract_nll_from_results(zero_shot_basic)
        if zero_shot_nll is not None:
            print(f"   ✅ Found zero-shot results: {zero_shot_nll:.4f} ({zero_shot_samples} samples)")
    elif Path(zero_shot_comprehensive).exists():
        zero_shot_nll, zero_shot_samples = extract_nll_from_results(zero_shot_comprehensive)
        if zero_shot_nll is not None:
            print(f"   ✅ Found zero-shot results (comprehensive): {zero_shot_nll:.4f} ({zero_shot_samples} samples)")
    
    if zero_shot_nll is None:
        print(f"   ❌ No zero-shot results found")
    
    # Check for history-only results
    history_only_file = f"context_free_eval/{dataset_name}_context_free_results.json"
    
    history_only_nll = None
    history_only_samples = 0
    
    if Path(history_only_file).exists():
        history_only_nll, history_only_samples = extract_nll_from_results(history_only_file)
        if history_only_nll is not None:
            print(f"   ✅ Found history-only results: {history_only_nll:.4f} ({history_only_samples} samples)")
    
    if history_only_nll is None:
        print(f"   ❌ No history-only results found")
    
    # Create visualization if we have at least one result OR baseline data exists
    baselines = get_dataset_baselines(dataset_name)
    if zero_shot_nll is not None or history_only_nll is not None or baselines:
        comparison_data = create_comprehensive_plot(dataset_name, zero_shot_nll, history_only_nll)
        
        # Show quick analysis
        print(f"   📊 Quick Analysis:")
        if zero_shot_nll is not None and baselines:
            vs_original = zero_shot_nll - baselines['original_centaur']
            print(f"      Zero-Shot vs Original: {vs_original:+.3f} ({'worse' if vs_original > 0 else 'better'})")
        
        if history_only_nll is not None and baselines:
            vs_original = history_only_nll - baselines['original_centaur']
            print(f"      History-Only vs Original: {vs_original:+.3f} ({'worse' if vs_original > 0 else 'better'})")
        
        if zero_shot_nll is not None and history_only_nll is not None:
            method_gap = abs(zero_shot_nll - history_only_nll)
            better = "Zero-Shot" if zero_shot_nll < history_only_nll else "History-Only"
            print(f"      Better method: {better} (gap: {method_gap:.3f})")
        
        return True
    else:
        print(f"   ❌ No evaluation results or baselines found for {dataset_name}")
        return False

def main():
    """Main function to process all available datasets"""
    print("=== Visualizing All Available Datasets ===")
    print("Creating comprehensive comparison plots for all tasks")
    print()
    
    # Create output directory
    output_dir = "all_datasets_plots"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # Find all available datasets
    datasets = find_available_datasets()
    print(f"🔍 Found {len(datasets)} datasets to process:")
    for dataset in datasets:
        print(f"   - {dataset}")
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for dataset in datasets:
        try:
            if process_dataset(dataset):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Error processing {dataset}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful} datasets")
    print(f"❌ Failed to process: {failed} datasets")
    print(f"📊 All plots saved to: {output_dir}/")
    
    if successful > 0:
        print(f"\n📋 Generated plots:")
        for file in sorted(output_path.glob("*_comprehensive_comparison.jpg")):
            print(f"   - {file.name}")
    
    print("\n🎉 Visualization complete!")

if __name__ == "__main__":
    main() 