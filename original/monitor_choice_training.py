#!/usr/bin/env python3
"""
Monitor training progress for choice prediction task.
Tracks task-specific metrics rather than general language modeling.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

def parse_training_logs(log_dir):
    """Parse training logs and extract key metrics."""
    log_files = list(Path(log_dir).glob("*.log"))
    
    metrics = {
        'step': [],
        'loss': [],
        'eval_loss': [],
        'learning_rate': []
    }
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if 'loss' in line and 'step' in line:
                    # Parse training metrics
                    step_match = re.search(r'step.*?(\d+)', line)
                    loss_match = re.search(r'loss.*?([0-9.]+)', line)
                    
                    if step_match and loss_match:
                        metrics['step'].append(int(step_match.group(1)))
                        metrics['loss'].append(float(loss_match.group(1)))
    
    return metrics

def check_convergence(metrics, patience=5):
    """Check if training has converged based on eval loss."""
    if len(metrics['eval_loss']) < patience:
        return False, "Not enough evaluation points"
    
    recent_losses = metrics['eval_loss'][-patience:]
    
    # Check if loss is plateauing (less than 1% improvement)
    if len(recent_losses) >= 2:
        improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        if improvement < 0.01:  # Less than 1% improvement
            return True, f"Loss plateaued: {improvement:.4f} improvement in last {patience} evals"
    
    return False, "Training still improving"

def plot_training_progress(metrics, save_path=None):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss
    ax1.plot(metrics['step'], metrics['loss'], label='Train Loss', alpha=0.7)
    if metrics['eval_loss']:
        # Assuming eval steps are different, we need to align them
        ax1.plot(metrics['eval_step'], metrics['eval_loss'], 
                label='Eval Loss', marker='o', alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    if metrics['learning_rate']:
        ax2.plot(metrics['step'], metrics['learning_rate'])
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Monitor choice prediction training')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='./monitoring_output',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Parse metrics
    print("Parsing training logs...")
    metrics = parse_training_logs(args.log_dir)
    
    if not metrics['step']:
        print("No training metrics found in logs!")
        return
    
    # Check convergence
    converged, message = check_convergence(metrics)
    print(f"\nConvergence Status: {message}")
    
    # Generate report
    Path(args.output_dir).mkdir(exist_ok=True)
    
    report = {
        'total_steps': max(metrics['step']) if metrics['step'] else 0,
        'final_loss': metrics['loss'][-1] if metrics['loss'] else None,
        'final_eval_loss': metrics['eval_loss'][-1] if metrics['eval_loss'] else None,
        'converged': converged,
        'convergence_message': message
    }
    
    with open(f"{args.output_dir}/training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Plot progress
    plot_training_progress(metrics, f"{args.output_dir}/training_progress.png")
    
    print(f"\nTraining Report:")
    print(f"Total Steps: {report['total_steps']}")
    print(f"Final Loss: {report['final_loss']:.4f}" if report['final_loss'] else "N/A")
    print(f"Final Eval Loss: {report['final_eval_loss']:.4f}" if report['final_eval_loss'] else "N/A")
    print(f"Converged: {report['converged']}")

if __name__ == "__main__":
    main()

"""
Standard Signs for Sufficient Training (Choice Prediction Task):

1. LOSS CONVERGENCE:
   - Training loss plateaus (< 1% improvement over 5 evaluations)
   - Eval loss stops decreasing or starts increasing (overfitting)

2. TASK-SPECIFIC METRICS:
   - Choice prediction accuracy on held-out experiments reaches plateau
   - Cross-entropy loss on choice tokens specifically (not all tokens)

3. BEHAVIORAL ALIGNMENT:
   - Model choices start matching human choice distributions
   - Performance on different experiment types stabilizes

4. EARLY STOPPING CRITERIA:
   - Eval loss increases for 3-5 consecutive evaluations (overfitting)
   - Training loss < 0.1 and not improving
   - Model starts predicting same choice for all contexts (mode collapse)

Usage:
python monitor_choice_training.py --log_dir ./centaur-random-init/logs
""" 