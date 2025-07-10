#!/usr/bin/env python3
"""
Complete workflow to evaluate your pre-trained model and generate comparison plots.

Usage:
    python evaluate_and_plot_comparison.py --model_path ./centaur-random-init
"""

import argparse
import subprocess
import os
import pandas as pd

def run_evaluation(model_path):
    """Run all evaluation scripts on the trained model."""
    print(f"🧠 Evaluating model: {model_path}")
    
    # Run main evaluation
    print("📊 Running main evaluation (test_adapter.py)...")
    subprocess.run([
        "python", "test_adapter.py", 
        "--model", model_path
    ], check=True)
    
    # Run custom metrics evaluation  
    print("📈 Running custom metrics evaluation...")
    subprocess.run([
        "python", "test_adapter_custom_metrics.py",
        "--model", model_path
    ], check=True)
    
    # Run full log likelihoods evaluation
    print("📉 Running full log likelihoods evaluation...")
    subprocess.run([
        "python", "test_adapter_full_log_likelihoods.py",
        "--model", model_path
    ], check=True)
    
    print("✅ Evaluation complete!")

def generate_comparison_plots(model_name):
    """Generate comparison plots between models."""
    print("📊 Generating comparison plots...")
    
    # Check if results file exists
    results_file = f"results/all_data_{model_name.replace('/', '-')}.csv"
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Make sure evaluation completed successfully.")
        return
    
    # Run modified plotting script
    os.chdir("plots")
    subprocess.run([
        "python", "fig2_pretrained_comparison.py"
    ], check=True)
    os.chdir("..")
    
    print("✅ Plots generated in plots/figures/")

def compare_performance(model_name):
    """Print performance comparison summary."""
    print("\n📋 Performance Summary:")
    print("=" * 50)
    
    try:
        # Load results
        df_pretrained = pd.read_csv(f"results/all_data_{model_name.replace('/', '-')}.csv")
        df_centaur = pd.read_csv("results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv")
        df_baseline = pd.read_csv("results/all_data_baseline.csv")
        df_random = pd.read_csv("results/all_data_random.csv")
        
        # Filter to participants (not experiments)
        df_pretrained = df_pretrained[df_pretrained['unseen'] == 'participants']
        df_centaur = df_centaur[df_centaur['unseen'] == 'participants']  
        df_baseline = df_baseline[df_baseline['unseen'] == 'participants']
        df_random = df_random[df_random['unseen'] == 'participants']
        
        # Calculate average pseudo-R²
        def calc_pseudo_r2(model_df, model_col, random_df):
            merged = pd.merge(model_df[['task', model_col]], 
                            random_df[['task', 'random']], on='task')
            ll_model = -merged[model_col]
            ll_random = -merged['random']
            return (1 - (ll_model/ll_random)).mean()
        
        r2_pretrained = calc_pseudo_r2(df_pretrained, model_name, df_random)
        r2_centaur = calc_pseudo_r2(df_centaur, 'marcelbinz/Llama-3.1-Centaur-70B-adapter', df_random)
        r2_baseline = calc_pseudo_r2(df_baseline, 'baseline', df_random)
        
        print(f"🤖 Pre-trained Centaur (yours): {r2_pretrained:.3f}")
        print(f"🦙 Original Centaur (fine-tuned): {r2_centaur:.3f}")
        print(f"🧠 Cognitive Models: {r2_baseline:.3f}")
        print(f"📈 Improvement over cognitive models: {r2_pretrained - r2_baseline:+.3f}")
        print(f"📊 Vs. original Centaur: {r2_pretrained - r2_centaur:+.3f}")
        
    except Exception as e:
        print(f"❌ Could not load results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare pre-trained model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to your trained model (e.g., ./centaur-random-init)')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation if already done')
    parser.add_argument('--plot_only', action='store_true', 
                       help='Only generate plots, skip evaluation')
    
    args = parser.parse_args()
    
    model_name = os.path.basename(args.model_path)
    
    if not args.plot_only and not args.skip_eval:
        run_evaluation(args.model_path)
    
    if not args.skip_eval:
        generate_comparison_plots(model_name)
    
    compare_performance(model_name)
    
    print("\n🎉 Complete! Check plots/figures/ for comparison visualizations.")

if __name__ == "__main__":
    main() 