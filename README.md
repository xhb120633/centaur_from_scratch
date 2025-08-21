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

Implementation details:
- **Per-trial forward pass**: Each trial is evaluated independently with a dedicated forward pass.
- **Choice-token NLL**: We locate the choice span (between `<<` and `>>`), tokenize it without special tokens, and compute the negative log-likelihood (NLL) by summing token-wise cross-entropy from the model's logits at the corresponding positions. The choice span includes all tokens within the delimiters (not just the first token).
- **Averaging policy**: Report per-trial NLL as the average across the choice tokens; aggregate across trials by mean.
- **Robustness**: If a choice span cannot be aligned reliably to token positions, the trial is skipped and reported.

#### 2. Context-Free (History-Only) Evaluation  
```python
# Original: Full task description + behavioral history
"Task: Choose between temporal rewards... You chose X. You chose Y. You choose <<X>>"

# Context-Free: Behavioral patterns only
"You chose X. You chose Y. You choose <<X>>"
```

Implementation details:
- **Single forward pass per participant**: One long prompt per participant containing all prior choices; we only extract likelihoods for marked choice spans.
- **No per-trial forward passes**: All per-trial (per-choice) NLLs are derived from the same forward pass' logits using sequential probing; we do not re-run the model for each trial.
- **Progressive NLL extraction**: While building prompts, we record exact character offsets for each upcoming `<<choice>>` span. At evaluation time:
  - Tokenize the entire prompt once and run a single forward pass to obtain logits.
  - For each recorded choice, search locally around the expected offset to find the exact `<<choice>>` substring.
  - Tokenize the substring (no special tokens) and align to the sequence positions to slice the corresponding logits window.
  - Compute token-wise log-probabilities via `log_softmax`, then NLL per token as `-log_prob[token_id]` and average across tokens for the trial NLL.
- **Aggregation**: Combine token-wise NLLs across trials for a participant, then across participants; report both overall NLL and counts of valid trials.
- **Efficiency**: This avoids repeated passes and mirrors the original evaluation logic while removing task context.
- **Multi-token choices**: The `<<choice>>` span is scored across all tokens inside the delimiters (not only the first token), using teacher-forced next-token likelihoods.

### Explicit Modes: Zero-Shot vs History-Only
- **Zero-Shot**:
  - Keeps task instructions/context; removes behavioral history
  - Per-trial forward pass with choice-token NLL per trial
  - Aggregate by mean across trials
- **History-Only (Context-Free)**:
  - Keeps behavioral history; removes task instructions/stimulus content
  - Single forward pass per participant with sequential probing of marked `<<choice>>` spans
  - Progressive, token-position-aligned NLLs aggregated across trials and participants

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

### Dataset acquisition
- **Psych-101 test set**: We obtain the test split via the Hugging Face dataset repository `marcelbinz/Psych-101-test`. Access requires agreeing to the repository terms.
  - Repository page: [Hugging Face: marcelbinz/Psych-101-test](https://huggingface.co/datasets/marcelbinz/Psych-101-test)
  - Download helper:
    ```bash
    # (Optional) set HF_TOKEN after accepting access on the dataset page
    setx HF_TOKEN "<your_hf_token>"

    # Download to datasets_downloads/ using our helper script
    python scripts/download_hf_dataset.py marcelbinz/Psych-101-test --local-dir datasets_downloads
    ```
- **Task structure via `original/experiments.csv`**: We use `original/experiments.csv` to map each dataset to its task name, type, split, and number of actions. This metadata helps us reorganize files from the Hugging Face repository so each task is clearly separated (train/validation/test) and matched to our evaluation routines.
- **Inspecting JSONL**: Use `scripts/inspect_jsonl.py datasets_downloads/prompts_testing_t1.jsonl` to preview schema, keys, and examples for quick sanity checks.

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

Concretely, we:
- Create a participant-level prompt and record `choice_positions` (expected character offsets for each `<<choice>>`).
- Tokenize once, compute logits, and for each choice: locate the substring, tokenize it without special tokens, align its token window, and compute token-wise NLLs via `log_softmax` followed by negation.
- Average per choice (trial) and aggregate across all trials and participants.

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

@misc{xie2025centaur_shortcut,
  title={Centaur May Have Learned a Shortcut that Explains Away Psychological Tasks},
  author={Xie, Hanbo and Zhu, Jian-Qiao},
  year={2025},
  month={July},
  publisher={PsyArXiv},
  doi={10.31234/osf.io/u7z4t_v1},
  url={https://doi.org/10.31234/osf.io/u7z4t_v1}
}
```

## 🏆 Acknowledgments

This fair evaluation framework builds upon the excellent Centaur research by Binz et al. We aim to provide rigorous, unbiased assessment of cognitive modeling capabilities while preserving the core insights of the original work.

**Data**: Original psychology experiments from the Centaur dataset and Psych-101 test set.  
**Infrastructure**: Evaluation pipeline adapted from the original Centaur codebase.  
**Methodology**: Novel fair evaluation approaches designed to address potential unfair advantages.

---

**🔗 Data Availability**: Complete results, datasets, and supplementary materials available on OSF: [OSF Project Link](https://osf.io/9nz76/) 
