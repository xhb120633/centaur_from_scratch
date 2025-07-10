# Centaur Pretraining from Random Initialization

A clean, modular codebase for training Centaur models from random initialization, built on top of the original Centaur research code.

## 🌟 Overview

This project allows you to:

1. **Train** Centaur models from random weights (rather than fine-tuning from pretrained Llama)
2. **Evaluate** models using the original evaluation scripts  
3. **Compare** your models against original Centaur and cognitive baselines
4. **Visualize** results with publication-quality plots

## 📁 Project Structure

```
├── src/                    # Clean modular source code
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation utilities  
│   ├── plotting/          # Plotting utilities
│   └── utils/             # Configuration and utilities
├── original/              # Original research codebase
├── configs/               # Configuration files
├── results/               # Training outputs and results
├── logs/                  # Training logs
├── plots/                 # Generated plots
└── run_experiment.py      # Main entry point
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies (same as original)
pip install unsloth transformers datasets torch trl
pip install scienceplots seaborn matplotlib pandas
```

### 2. Full Pipeline (Train + Evaluate + Plot)

```bash
# Run complete experiment with defaults
python run_experiment.py --full

# Custom experiment
python run_experiment.py --full --epochs 25 --learning_rate 1e-4 --name "my_experiment"
```

### 3. Individual Components

```bash
# Training only
python run_experiment.py --train --epochs 10 --learning_rate 2e-4

# Evaluate existing model
python run_experiment.py --evaluate --model results/my-model

# Create comparison plots
python run_experiment.py --plot --model results/my-model
```

## ⚙️ Configuration

### Using Config Files

```bash
# Create default config
python run_experiment.py --create_config

# Use custom config
python run_experiment.py --full --config configs/my_config.json
```

### Command Line Overrides

```bash
python run_experiment.py --train \
    --epochs 50 \
    --learning_rate 1e-4 \
    --batch_size 2 \
    --output_dir results/my-experiment \
    --name "centaur_50_epochs" \
    --description "Testing longer training"
```

## 🧪 Example Configurations

### Quick Test (5 epochs)
```bash
python run_experiment.py --full --config configs/quick_test.json
```

### Full Training (25 epochs - recommended)
```bash
python run_experiment.py --full --epochs 25 --learning_rate 1e-4
```

### Fine-tuning from Pretrained (comparison)
```bash
# Create config with random_init: false
python run_experiment.py --train --config configs/finetune_config.json
```

## 📊 Evaluation & Results

The evaluation automatically runs:

1. **Main Tasks** (36 psychology experiments) - `test_adapter.py`
2. **Custom Metrics** (10 held-out tasks) - `test_adapter_custom_metrics.py`
3. **Full Log Likelihoods** - `test_adapter_full_log_likelihoods.py`

### Understanding Results

- **Pseudo-R²**: `1 - (model_loss / random_loss)` - higher is better
- **Main Tasks**: Performance on same experiments, different participants
- **Held-out Tasks**: Performance on completely unseen experiments

## 📈 Plotting & Visualization

### Automatic Comparison Plots

The plotting module creates publication-ready figures comparing:

- 🤖 **Your Pretrained Model** (trained from random)
- 🦙 **Original Centaur** (fine-tuned from Llama)  
- 🧠 **Cognitive Models** (domain-specific baselines)

### Example Output

```
📊 Performance Summary:
🤖 Centaur-Random-Init: 0.245
🦙 Original Centaur: 0.387
🧠 Cognitive Models: 0.298
📈 Your Model vs Cognitive Models: -0.053
📊 Your Model vs Original Centaur: -0.142
```

## 🔧 Technical Details

### Model Architecture
- **Base**: Llama 3.1-70B with random initialization
- **Adaptation**: LoRA (Low-Rank Adaptation)
- **Training**: Supervised fine-tuning on psychology tasks
- **Masking**: Only trains on choice tokens (`<<choice>>`)

### Key Differences from Original
- Starts from **random weights** instead of pretrained Llama
- **Modular architecture** for easy experimentation
- **Clean configuration system** for reproducibility
- **Automated evaluation pipeline** 

### Training Settings
```python
# Default configuration
learning_rate: 1e-4      # Higher than fine-tuning (5e-5)
num_epochs: 25           # More than fine-tuning (5)  
warmup_steps: 1000       # More warmup for random init
batch_size: 1            # Same as original
gradient_accumulation: 32 # Same as original
```

## 🎯 Expected Results

Based on our understanding of the task:

1. **Random initialization will perform worse** than fine-tuning from pretrained
2. **But should still beat random baselines** and potentially some cognitive models
3. **Training from scratch requires more epochs** (25+ vs 5 for fine-tuning)
4. **Performance gap quantifies the value** of language pretraining for psychology

## 🛠️ Advanced Usage

### Custom Training Scripts

```python
from src.training.train_from_random import CentaurTrainer
from src.utils.config import TrainingConfig

config = TrainingConfig()
config.num_epochs = 50
config.learning_rate = 2e-4

trainer = CentaurTrainer(config)
model_path = trainer.run_full_pipeline()
```

### Custom Evaluation

```python
from src.evaluation.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model("results/my-model")
evaluator.print_summary("my-model")
```

### Custom Plotting

```python
from src.plotting.compare_models import ModelComparator

comparator = ModelComparator()
plot_path = comparator.create_main_comparison_plot("results/my-model")
```

## 🤝 Integration with Original Codebase

This codebase is designed to seamlessly use the original research code:

- **Imports evaluation scripts** from `original/` directory
- **Uses same datasets** (Psych-101, Psych-101-test)
- **Compatible result formats** for plotting
- **Leverages existing infrastructure** while providing clean interface

## 📋 TODO / Extensions

- [ ] Multi-GPU training support
- [ ] Hyperparameter optimization
- [ ] Additional model sizes (8B, other architectures)
- [ ] Comparison with other pretrained models
- [ ] Analysis of which psychology domains benefit most from pretraining

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Import errors**: Ensure `original/` directory contains the research code
3. **Missing results**: Run evaluation before plotting
4. **Permission errors**: Check file permissions in results/ and logs/

### Getting Help

```bash
# Check configuration
python run_experiment.py --create_config

# Validate setup
python -c "from src.training.train_from_random import CentaurTrainer; print('✅ Setup OK')"

# Debug training
python run_experiment.py --train --config configs/quick_test.json
```

## 📄 Citation

If you use this codebase, please cite both the original Centaur paper and acknowledge this implementation:

```bibtex
@article{binz2024centaur,
  title={Centaur: a foundation model of human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and Brändle, Franziska and Callaway, Fred and Gijsbers, Pieter and Gökce, Ozan and Gronau, Quentin F and Lampinen, Andrew K and Lawson, Tobias and others},
  journal={arXiv preprint arXiv:2410.20268},
  year={2024}
}
```

## 🏆 Acknowledgments

This implementation builds upon the excellent work by Binz et al. in the original Centaur research. We provide a clean, modular interface while preserving all the core functionality and evaluation capabilities of the original codebase. 