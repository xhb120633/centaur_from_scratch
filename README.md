# Fair Evaluation of Centaur: Zero-Shot and Context-Free Analysis

A comprehensive evaluation framework testing whether Centaur's performance relies on unfair advantages or captures genuine cognitive patterns through fair zero-shot and context-free methodologies.

## 🎯 Overview

This project provides **fair evaluation** of the Centaur foundation model by removing potential unfair advantages:

1. **Zero-Shot Evaluation**: Tests Centaur without behavioral history (like humans in first encounters)
2. **Context-Free Evaluation**: Tests whether Centaur captures behavioral patterns without task-specific context
3. **Statistical Comparison**: Rigorous comparison against cognitive models and random baselines
4. **Multi-Dataset Analysis**: Evaluation across diverse psychology experiments

### Key Question
**Does Centaur truly capture human cognitive patterns, or does it rely on unfair advantages like behavioral history and task-specific context?**

## 📊 Methodology

### Original Centaur (Potentially Unfair)
- Uses **full behavioral history** from previous trials
- Includes **complete task context** and instructions
- May benefit from **in-context learning** rather than cognitive modeling

### Our Fair Evaluations

#### 1. Zero-Shot Evaluation
```python
# Original: Full history + context
"You choose between X and Y. You chose X. You chose Y. You chose X. You choose <<X>>"

# Zero-Shot: Individual trials only  
"You choose between X and Y. You choose <<X>>"
```

#### 2. Context-Free Evaluation  
```python
# Original: Full task description + behavioral history
"Task: Choose between temporal rewards... You chose X. You chose Y. You choose <<X>>"

# Context-Free: Behavioral patterns only
"You chose X. You chose Y. You choose <<X>>"
```

## 📁 Project Structure

```
├── evaluate_zero_shot_centaur.py     # Fair zero-shot evaluation
├── evaluate_history_only_centaur.py  # Context-free evaluation  
├── visualize_all_datasets.py         # Comprehensive plotting
├── collect_trial_level_nlls.py       # Statistical analysis
├── statistical_analysis.py           # Significance testing
├── original/                         # Original Centaur codebase
├── configs/                          # Evaluation configurations
├── all_datasets_plots/              # Generated visualizations
├── statistical_tests/               # Statistical test results
└── requirements.txt                 # Dependencies
```

## 🗂️ Data Availability

**Results and datasets are available on OSF**: [[OSF Project Link](https://osf.io/9nz76/)]

Due to size constraints, the following directories are stored externally:
- `eval_results/` - Zero-shot evaluation results  
- `context_free_eval/` - Context-free evaluation results
- `test_datasets/` - Psychology experiment datasets (Psych-101 test set)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Additional for plotting (optional)
pip install scienceplots
```

### 2. Run Fair Evaluations

#### Zero-Shot Evaluation
```bash
# Evaluate specific dataset
python evaluate_zero_shot_centaur.py --task hebart2023things

# List available datasets
python evaluate_zero_shot_centaur.py --list-datasets

# Large dataset with efficient processing
python evaluate_zero_shot_centaur.py --task dubois2022value --skip-detailed-analysis
```

#### Context-Free Evaluation
```bash
# Evaluate with behavioral patterns only
python evaluate_history_only_centaur.py --task ruggeri2022globalizability

# Multiple tasks supported
python evaluate_history_only_centaur.py --task hilbig2014generalized
python evaluate_history_only_centaur.py --task hebart2023things
```

### 3. Visualization and Analysis

```bash
# Generate comparison plots for all datasets
python visualize_all_datasets.py

# Collect trial-level data for statistical testing
python collect_trial_level_nlls.py

# Run statistical significance tests
python statistical_analysis.py
```

## 📋 Supported Datasets

| Dataset | Task Type | Choices | Description |
|---------|-----------|---------|-------------|
| `ruggeri2022globalizability` | Temporal Decision | Binary | Intertemporal choice task |
| `hilbig2014generalized` | Multi-attribute | Binary | Decision-making with expert ratings |
| `hebart2023things` | Odd-one-out | 3-choice | THINGS object similarity task |
| `collsioo2023MCPL_all` | Sequential Learning | 9-choice | Multi-cue judgment task |
| `wu2018generalisation_exp1` | Exploration | 30-choice | Multi-armed bandit task |
| `dubois2022value` | Sequential Value | Binary | Value-based decision making |

## 🎨 Visualization

### Automatic Comparison Plots

The visualization generates publication-ready square plots comparing:

- 🟦 **Original Centaur** (potentially unfair)
- 🟪 **Cognitive Models** (Hyperbolic discounting, GP-UCB, etc.)  
- 🟧 **Centaur without Psychological Task** (context-free)
- 🟩 **Zero-Shot Centaur** (fair evaluation)
- ⚫ **Random Guessing** (baseline)

### Example Usage
```bash
python visualize_all_datasets.py
# Generates: all_datasets_plots/[dataset]_comprehensive_comparison.png/pdf
```

## 📊 Statistical Analysis

### Trial-Level Analysis
```bash
python collect_trial_level_nlls.py
# Outputs:
# - trial_level_nlls_all_methods.npz (numpy arrays)
# - trial_level_nlls_long_format.csv (for statistical testing)
# - trial_level_nlls_metadata.json (dataset information)
```

### Significance Testing
```bash
python statistical_analysis.py
# Generates:
# - statistical_tests/[dataset]_statistical_analysis.json
# - Wilcoxon signed-rank tests
# - Effect size calculations
# - Bonferroni corrections
```

## 🔬 Key Findings

### Expected Results

1. **Zero-Shot < Original Centaur**: Fair evaluation should perform worse than unfair
2. **Context-Free ≈ Cognitive Models**: Pure behavioral patterns vs domain expertise  
3. **Both > Random**: Even fair evaluations should beat chance
4. **Performance Gaps**: Quantify the unfair advantage

### Research Questions Addressed

- ✅ **Does Centaur rely on behavioral history?** (Zero-shot evaluation)
- ✅ **Does Centaur capture cognitive patterns?** (Context-free evaluation)  
- ✅ **How much advantage comes from context?** (Statistical comparison)
- ✅ **Is performance genuine or artifactual?** (Fair vs unfair comparison)

## ⚙️ Advanced Configuration

### Custom Evaluation Parameters

```bash
# Adjust batch size for memory constraints
python evaluate_zero_shot_centaur.py --task hebart2023things --batch-size 2

# Skip detailed analysis for large datasets  
python evaluate_zero_shot_centaur.py --task dubois2022value --skip-detailed-analysis

# Use KV caching optimization
python evaluate_zero_shot_centaur.py --task dubois2022value --use-kv-caching
```

### Model Configuration

The evaluations use the pre-trained Centaur model:
- **Model**: `marcelbinz/Llama-3.1-Centaur-70B-adapter`
- **Tokenizer**: Same as base model
- **Inference**: 4-bit quantization for efficiency
- **Context Length**: 32,768 tokens

## 🧪 Evaluation Pipeline

### 1. Dataset-Specific Parsing
Each dataset requires custom parsing to extract fair prompts:

```python
# Example for THINGS dataset
def create_zero_shot_prompts_hebart(original_file, output_file):
    # Extract: "A: object1, B: object2, C: object3. You press <<A>>."
    # Convert to: "You press <<A>>." (individual trials)
```

### 2. Progressive NLL Extraction
For behavioral history evaluation:
```python
# Single forward pass per participant
# Extract NLL for each choice token progressively
# Maintains computational efficiency
```

### 3. Statistical Comparison
```python
# Compare distributions using:
# - Wilcoxon signed-rank test (non-parametric)
# - Effect size calculation (Cohen's d)
# - Multiple comparison correction (Bonferroni)
```

## 🔧 Technical Details

### Memory Optimization
- **4-bit quantization** reduces memory usage
- **Batch processing** for large datasets
- **Progressive NLL extraction** avoids redundant computation
- **KV caching** for sequential tasks

### Computational Requirements
- **GPU**: H100/H200 recommended (70B model)
- **Memory**: ~40GB base + 15GB per batch unit
- **Time**: ~1-3 hours per dataset (varies by size)

## 🤝 Integration with Original Codebase

This evaluation framework:
- **Preserves original evaluation logic** (same metrics, same baselines)
- **Uses identical model weights** (marcelbinz/Llama-3.1-Centaur-70B-adapter)
- **Maintains compatibility** with original result formats
- **Only changes prompt structure** for fair evaluation

## 📋 Reproducibility

### Configuration Files
```bash
# Example: configs/quick_test.json
{
  "model_name": "marcelbinz/Llama-3.1-Centaur-70B-adapter",
  "batch_size": 4,
  "max_seq_length": 32768,
  "quantization": "4bit"
}
```

### Seed Management
```python
# Consistent random seeds for reproducible results
# Statistical noise generation uses fixed seeds
# Ensures reproducible fair evaluations
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   python evaluate_zero_shot_centaur.py --task [dataset] --batch-size 1
   ```

2. **Missing datasets**
   - Download from OSF project (link above)
   - Place in appropriate directories

3. **Slow evaluation**
   ```bash
   python evaluate_zero_shot_centaur.py --skip-detailed-analysis
   ```

4. **Plot overlapping labels**
   - Automatically handled with 45° rotation and size adjustment

### Debug Mode
```bash
# Test with small subset
python evaluate_zero_shot_centaur.py --task ruggeri2022globalizability --batch-size 1
```

## 📄 Citation

```bibtex
@article{binz2024centaur,
  title={Centaur: a foundation model of human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and others},
  journal={arXiv preprint arXiv:2410.20268},
  year={2024}
}

@misc{centaur_fair_evaluation,
  title={Fair Evaluation of Centaur: Zero-Shot and Context-Free Analysis},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub repository and OSF project}
}
```

## 🏆 Acknowledgments

This fair evaluation framework builds upon the excellent Centaur research by Binz et al. We aim to provide rigorous, unbiased assessment of cognitive modeling capabilities while preserving the core insights of the original work.

**Data**: Original psychology experiments from the Centaur dataset and Psych-101 test set.  
**Infrastructure**: Evaluation pipeline adapted from the original Centaur codebase.  
**Methodology**: Novel fair evaluation approaches designed to address potential unfair advantages.

---

**🔗 Data Availability**: Complete results, datasets, and supplementary materials available on OSF: [OSF Project Link](https://osf.io/9nz76/) 
