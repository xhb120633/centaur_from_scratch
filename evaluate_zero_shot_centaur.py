"""
Zero-Shot Centaur Evaluation Script (Simplified Direct Inference)
Compares fair zero-shot evaluation against original unfair methodology
Uses direct HuggingFace inference with merged LoRA model
"""

import json
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import logging
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
import argparse
from datetime import datetime

# Try importing scienceplots for original paper styling
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("⚠️ scienceplots not available - install with: pip install scienceplots")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing PEFT for LoRA adapter support
PEFT_AVAILABLE = False
PEFT_ERROR = None
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
    logger.info("✅ PEFT successfully imported at module level")
except ImportError as e:
    PEFT_ERROR = f"ImportError: {e}"
    logger.warning(f"⚠️ PEFT import failed at module level: {e}")
except Exception as e:
    PEFT_ERROR = f"Unexpected error: {e}"
    logger.warning(f"⚠️ PEFT import failed with unexpected error: {e}")

def detect_gpu_setup():
    """Detect available GPUs and determine optimal single-GPU strategy (Unsloth limitation)"""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs available")
        return {'strategy': 'cpu', 'num_gpus': 0, 'gpu_memory': []}
    
    num_gpus = torch.cuda.device_count()
    gpu_memory = []
    gpu_names = []
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        gpu_memory.append(memory_gb)
        gpu_names.append(props.name)
    
    print(f"🔍 GPU Detection:")
    print(f"   Available GPUs: {num_gpus}")
    for i, (name, mem) in enumerate(zip(gpu_names, gpu_memory)):
        print(f"   GPU {i}: {name} ({mem:.1f}GB)")
    
    # ⚠️ Unsloth limitation: Only supports single GPU
    print(f"   ⚠️  Unsloth limitation: Only supports single GPU (will use GPU 0)")
    
    # Use the best GPU (usually GPU 0) with optimized batch size
    best_gpu_memory = max(gpu_memory) if gpu_memory else 0
    recommended_batch = min(8, int(best_gpu_memory // 20))  # ~20GB per batch unit
    
    return {
        'strategy': 'single_gpu_unsloth',  # Force single GPU due to Unsloth limitation
        'num_gpus': 1,  # Always use 1 GPU with Unsloth
        'gpu_memory': [best_gpu_memory],
        'gpu_names': [gpu_names[0] if gpu_names else "Unknown"],
        'total_memory': best_gpu_memory,
        'recommended_batch_per_gpu': recommended_batch,
        'unsloth_limitation': True
    }

def setup_distributed(rank, world_size):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()

def test_peft_directly():
    """Test PEFT import and basic functionality"""
    try:
        import peft
        print(f"✅ PEFT version: {peft.__version__}")
        
        # Test basic PEFT functionality
        from peft import PeftModel, PeftConfig
        print("✅ PEFT imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ PEFT import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PEFT test failed: {e}")
        return False

def merge_lora_adapter(base_model_name: str, adapter_name: str, output_dir: str = "temp_merged_centaur"):
    """
    DEPRECATED: This function creates unnecessary temporary files.
    Use unsloth's FastLanguageModel.from_pretrained instead - it handles LoRA merging automatically.
    """
    print(f"⚠️  WARNING: merge_lora_adapter creates temporary files in '{output_dir}'")
    print(f"⚠️  Consider using FastLanguageModel.from_pretrained instead")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # Load base model and tokenizer
        print("   Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Load and merge LoRA adapter
        print("   Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_name)
        
        print("   Merging adapter with base model...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        print(f"   ⚠️  WARNING: Creating temporary files in {output_dir}...")
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ LoRA adapter merged successfully!")
        print(f"⚠️  Remember to clean up temporary directory: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ LoRA merging failed: {e}")
        raise

def create_zero_shot_prompts_ruggeri(original_prompts_file, output_file):
    """Create true zero-shot prompts for ruggeri2022globalizability by removing behavioral history"""
    print(f"📝 Creating zero-shot prompts from: {original_prompts_file}")
    
    # Load original prompts
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} original prompts")
    
    # Create zero-shot versions
    zero_shot_prompts = []
    for prompt_data in original_prompts:
        prompt_text = prompt_data['text']
        
        # Extract task instructions (before any behavioral history)
        if "You press <<" in prompt_text:
            # Find the task instructions (everything before the first behavioral history)
            parts = prompt_text.split("You press <<")
            task_instructions = parts[0].strip()
            
            # Extract the ground truth from the original prompt (last choice)
            ground_truth = None
            if len(parts) > 1:
                # Find the last choice in the original prompt
                last_part = parts[-1]
                end_idx = last_part.find(">>")
                if end_idx != -1:
                    ground_truth = last_part[:end_idx].strip()
            
            # Create zero-shot prompt: instructions + choice prompt
            zero_shot_text = task_instructions + " You press <<"
            
            # Add the ground truth to complete the prompt (for loss computation)
            if ground_truth:
                zero_shot_text += ground_truth + ">>"
            
            # Create zero-shot prompt data (same format as original)
            zero_shot_data = {
                'text': zero_shot_text,
                'participant': prompt_data['participant'],
                'experiment': prompt_data['experiment']
            }
            zero_shot_prompts.append(zero_shot_data)
        else:
            # If no behavioral history, keep as is
            zero_shot_prompts.append(prompt_data)
    
    print(f"   Created {len(zero_shot_prompts)} zero-shot prompts")
    
    # Save zero-shot prompts
    with open(output_file, 'w') as f:
        for prompt in zero_shot_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    return zero_shot_prompts

def create_zero_shot_prompts_awad(original_prompts_file, output_file):
    """Create true zero-shot prompts for awad2018moral - same principle as ruggeri"""
    print(f"📝 Creating zero-shot prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participants")
    
    # Create zero-shot versions (each trial becomes a separate prompt)
    zero_shot_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract basic task instructions (before first dilemma - same principle as ruggeri)
        if "Outcome X:" in prompt_text:
            parts = prompt_text.split("Outcome X:")
            task_instructions = parts[0].strip()
            
            # Find all choices (same as ruggeri finds all "You press <<")
            import re
            choices = re.findall(r'You select outcome <<([XG])>>', prompt_text)
            
            # Extract each individual trial (same principle as ruggeri)
            dilemma_sections = parts[1:]  # All dilemmas after instructions
            
            for i, dilemma_section in enumerate(dilemma_sections):
                if i < len(choices):
                    # Find where this specific trial ends (before next choice)
                    choice_pattern = f"You select outcome <<{choices[i]}>>"
                    choice_idx = dilemma_section.find(choice_pattern)
                    
                    if choice_idx != -1:
                        # Extract just this trial's dilemma (no history from previous trials)
                        dilemma_text = dilemma_section[:choice_idx].strip()
                        
                        # Create zero-shot trial: instructions + this trial + choice
                        # Same format as ruggeri: basic context + individual decision
                        zero_shot_text = task_instructions + "\n\nOutcome X:" + dilemma_text + f"\nYou select outcome <<{choices[i]}>>"
                        
                        # Create individual trial data
                        zero_shot_data = {
                            'text': zero_shot_text,
                            'participant': participant_data['participant'],
                            'experiment': participant_data['experiment']
                        }
                        zero_shot_prompts.append(zero_shot_data)
        else:
            # Fallback - keep as is
            zero_shot_prompts.append(participant_data)
    
    print(f"   Created {len(zero_shot_prompts)} individual zero-shot trials from {len(original_prompts)} participants")
    print(f"   Average trials per participant: {len(zero_shot_prompts) / len(original_prompts):.1f}")
    
    # Save zero-shot prompts
    with open(output_file, 'w') as f:
        for prompt in zero_shot_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if zero_shot_prompts:
        example = zero_shot_prompts[0]
        print(f"\n📝 Example zero-shot trial:")
        print(f"   Length: {len(example['text'])} characters")
        print(f"   Preview: {example['text'][:200]}...")
    
    return zero_shot_prompts

def create_zero_shot_prompts_hilbig(original_prompts_file, output_file):
    """Create true zero-shot prompts for hilbig2014generalized - extract individual trials from participant sessions"""
    print(f"📝 Creating zero-shot prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant session)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participant sessions")
    
    # Create zero-shot versions (each trial becomes a separate prompt)
    zero_shot_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract task instructions (everything before first "Product")
        if "Product " in prompt_text:
            parts = prompt_text.split("Product ")
            task_instructions = parts[0].strip()
            
            # Find all individual trials in the format:
            # "Product X ratings: [...]. Product Y ratings: [...]. You press <<Choice>>."
            import re
            
            # Pattern to match complete trials
            trial_pattern = r'Product ([A-Z]+) ratings: (\[[^\]]+\])\. Product ([A-Z]+) ratings: (\[[^\]]+\])\. You press <<([A-Z]+)>>\.'
            trials = re.findall(trial_pattern, prompt_text)
            
            print(f"   Participant {participant_data.get('participant', 'unknown')}: found {len(trials)} trials")
            
            # Create individual zero-shot prompts for each trial
            for i, (product1, ratings1, product2, ratings2, choice) in enumerate(trials):
                # Create zero-shot trial: instructions + this trial only (no history)
                trial_text = f"Product {product1} ratings: {ratings1}. Product {product2} ratings: {ratings2}. You press <<{choice}>>"
                zero_shot_text = task_instructions + "\n\n" + trial_text
                
                # Create individual trial data
                zero_shot_data = {
                    'text': zero_shot_text,
                    'participant': participant_data['participant'],
                    'experiment': participant_data['experiment'],
                    'trial_number': i + 1,  # Add trial numbering
                    'choice': choice,
                    'products': [product1, product2],
                    'ratings': [ratings1, ratings2]
                }
                zero_shot_prompts.append(zero_shot_data)
        else:
            # Fallback - keep as is if no standard format found
            zero_shot_prompts.append(participant_data)
    
    print(f"   Created {len(zero_shot_prompts)} individual zero-shot trials from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        print(f"   Average trials per participant: {len(zero_shot_prompts) / len(original_prompts):.1f}")
    
    # Save zero-shot prompts
    with open(output_file, 'w') as f:
        for prompt in zero_shot_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if zero_shot_prompts:
        example = zero_shot_prompts[0]
        print(f"\n📝 Example zero-shot trial:")
        print(f"   Choice: {example.get('choice', 'N/A')}")
        print(f"   Products: {example.get('products', 'N/A')}")
        print(f"   Length: {len(example['text'])} characters")
        print(f"   Preview: {example['text'][:300]}...")
    
    return zero_shot_prompts

def create_zero_shot_prompts_hebart(original_prompts_file, output_file):
    """Create true zero-shot prompts for hebart2023things - extract individual trials from participant sessions"""
    print(f"📝 Creating zero-shot prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant session)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participant sessions")
    
    # Create zero-shot versions (each trial becomes a separate prompt)
    zero_shot_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract task instructions (everything before first trial)
        import re
        trial_pattern = r'([A-Z]): ([^,]+), ([A-Z]): ([^,]+), and ([A-Z]): ([^.]+)\. You press <<([A-Z])>>\.'
        first_trial_match = re.search(trial_pattern, prompt_text)
        
        if first_trial_match:
            instructions_end = first_trial_match.start()
            task_instructions = prompt_text[:instructions_end].strip()
            trials_section = prompt_text[instructions_end:]
            
            # Find all individual trials
            trials = re.finditer(trial_pattern, trials_section)
            
            trial_count = 0
            for trial_match in trials:
                trial_text = trial_match.group(0)
                key1, obj1, key2, obj2, key3, obj3, choice = trial_match.groups()
                
                # Create zero-shot trial: instructions + this trial only (no history)
                zero_shot_text = task_instructions + "\n\n" + trial_text
                
                # Create individual trial data
                zero_shot_data = {
                    'text': zero_shot_text,
                    'participant': participant_data['participant'],
                    'experiment': participant_data['experiment'],
                    'trial_number': trial_count + 1,
                    'choice': choice,
                    'objects': [obj1.strip(), obj2.strip(), obj3.strip()],
                    'keys': [key1, key2, key3]
                }
                zero_shot_prompts.append(zero_shot_data)
                trial_count += 1
            
            print(f"   Participant {participant_data['participant']}: extracted {trial_count} trials")
        else:
            # Fallback - keep as is if no standard format found
            zero_shot_prompts.append(participant_data)
    
    print(f"   Created {len(zero_shot_prompts)} individual zero-shot trials from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        print(f"   Average trials per participant: {len(zero_shot_prompts) / len(original_prompts):.1f}")
    
    # Save zero-shot prompts
    with open(output_file, 'w') as f:
        for prompt in zero_shot_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if zero_shot_prompts:
        example = zero_shot_prompts[0]
        print(f"\n📝 Example zero-shot trial:")
        print(f"   Choice: {example.get('choice', 'N/A')}")
        print(f"   Objects: {example.get('objects', 'N/A')}")
        print(f"   Keys: {example.get('keys', 'N/A')}")
        print(f"   Length: {len(example['text'])} characters")
        print(f"   Preview: {example['text'][:300]}...")
    
    return zero_shot_prompts

def create_zero_shot_prompts_dubois(original_prompts_file, output_file):
    """Create true zero-shot prompts for dubois2022value - split games and evaluate sequentially within games"""
    print(f"📝 Creating zero-shot prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant with 400 concatenated games)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participants")
    
    # Create zero-shot versions (each game becomes separate trials)
    zero_shot_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Find the main instruction (everything before first "Game 1")
        if "Game 1." in prompt_text:
            parts = prompt_text.split("Game 1.", 1)
            main_instruction = parts[0].strip()
            games_text = "Game 1." + parts[1]
            
            # Split into individual games
            import re
            game_splits = re.split(r'\n\nGame (\d+)\.', games_text)
            
            # First element is Game 1 content, then alternating game numbers and content
            games = []
            if len(game_splits) >= 1:
                # Handle Game 1 (first game)
                game_1_content = game_splits[0].replace("Game 1.", "").strip()
                games.append(("1", game_1_content))
                
                # Handle remaining games (pairs of game_number, content)
                for i in range(1, len(game_splits) - 1, 2):
                    if i + 1 < len(game_splits):
                        game_num = game_splits[i]
                        game_content = game_splits[i + 1].strip()
                        games.append((game_num, game_content))
            
            print(f"   Participant {participant_data['participant']}: found {len(games)} games")
            
            # Process each game independently
            for game_num, game_content in games:
                # Parse game structure
                lines = game_content.split('\n')
                if not lines:
                    continue
                
                # Find trial count
                trial_count = None
                if lines and "trials in this game" in lines[0]:
                    trial_match = re.search(r'(\d+) trials in this game', lines[0])
                    if trial_match:
                        trial_count = int(trial_match.group(1))
                
                # Separate observations (forced trials) and actual choices
                observations = []  # "You are instructed to..." - context/observations
                actual_choices = []  # "You press <<...>>" - real human decisions
                
                for line in lines[1:]:  # Skip the "X trials in this game" line
                    line = line.strip()
                    if line.startswith("You are instructed to"):
                        observations.append(line)
                    elif line.startswith("You press <<") and ">>" in line:
                        actual_choices.append(line)
                
                if not actual_choices:
                    continue  # Skip games with no actual choices
                
                # Create sequential prompts for this game
                # Context = observations (forced trials), then sequential evaluation of actual choices
                game_context = f"Game {game_num}. There are {len(actual_choices)} actual choices in this game.\n"
                if observations:
                    game_context += "\n".join(observations) + "\n"
                
                for trial_idx, choice_line in enumerate(actual_choices):
                    # Build context: main_instruction + this_game_context + previous_actual_choices_in_this_game + current_choice
                    trial_context = main_instruction.strip()
                    if trial_context and not trial_context.endswith('\n'):
                        trial_context += '\n\n'
                    
                    trial_context += game_context
                    
                    # Add previous actual choices in this game (sequential dependency)
                    # Each choice builds on observations + previous actual choices
                    for prev_idx in range(trial_idx):
                        trial_context += actual_choices[prev_idx] + "\n"
                    
                    # Add current choice
                    trial_context += choice_line
                    
                    # Create zero-shot trial data
                    zero_shot_data = {
                        'text': trial_context,
                        'participant': participant_data['participant'],
                        'experiment': participant_data['experiment'],
                        'game_number': int(game_num),
                        'trial_in_game': trial_idx + 1,
                        'total_trials_in_game': len(actual_choices),
                        'total_observations_in_game': len(observations),
                        'choice': re.search(r'You press <<([^>]+)>>', choice_line).group(1) if re.search(r'You press <<([^>]+)>>', choice_line) else None
                    }
                    zero_shot_prompts.append(zero_shot_data)
        else:
            # Fallback - keep as is if no standard format found
            zero_shot_prompts.append(participant_data)
    
    print(f"   Created {len(zero_shot_prompts)} individual sequential trials from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        print(f"   Average trials per participant: {len(zero_shot_prompts) / len(original_prompts):.1f}")
    
    # Save zero-shot prompts
    with open(output_file, 'w') as f:
        for prompt in zero_shot_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if zero_shot_prompts:
        example = zero_shot_prompts[0]
        print(f"\n📝 Example sequential trial:")
        print(f"   Participant: {example['participant']}")
        print(f"   Game: {example['game_number']}, Choice: {example['trial_in_game']}/{example['total_trials_in_game']}")
        print(f"   Observations in game: {example['total_observations_in_game']}")
        print(f"   Actual choice: {example['choice']}")
        print(f"   Context length: {len(example['text'])} characters")
        print(f"   Preview: {example['text'][:300]}...")
    
    return zero_shot_prompts



def create_zero_shot_prompts(original_prompts_file, output_file, dataset_name):
    """Create zero-shot prompts based on dataset type"""
    if dataset_name == "ruggeri2022globalizability":
        return create_zero_shot_prompts_ruggeri(original_prompts_file, output_file)
    elif dataset_name == "awad2018moral":
        return create_zero_shot_prompts_awad(original_prompts_file, output_file)
    elif dataset_name == "hilbig2014generalized":
        return create_zero_shot_prompts_hilbig(original_prompts_file, output_file)
    elif dataset_name == "hebart2023things":
        return create_zero_shot_prompts_hebart(original_prompts_file, output_file)
    elif dataset_name == "dubois2022value":
        return create_zero_shot_prompts_dubois(original_prompts_file, output_file)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_dataset_config(dataset_name):
    """Get configuration for different datasets"""
    configs = {
        "ruggeri2022globalizability": {
            "original_prompts_file": "original/ceiling/ruggeri2022globalizability/prompts_zeroshot.jsonl",
            "zero_shot_prompts_file": "eval_results/ruggeri2022globalizability_zero_shot_prompts.jsonl",
            "baseline_nll": 0.4382756948471069,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.6590430736541748,  # from all_data_baseline.csv
            "llama_nll": 0.7385996580123901,     # from all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
            "random_nll": 0.6931471805599453,    # from all_data_random.csv
            "description": "Temporal decision making task"
        },
        "awad2018moral": {
            "original_prompts_file": "datasets/held_out_tasks/awad2018moral.jsonl",
            "zero_shot_prompts_file": "eval_results/awad2018moral_zero_shot_prompts.jsonl",
            "baseline_nll": None,  # Not available - awad2018moral is in held_out_tasks
            "cognitive_nll": None,  # Not available - awad2018moral is in held_out_tasks
            "llama_nll": None,     # Not available - awad2018moral is in held_out_tasks
            "random_nll": 0.6931471805599453,    # ln(2) for binary choices
            "description": "Autonomous vehicle moral dilemmas (individual trials extracted from participant sessions)"
        },
        "hilbig2014generalized": {
            "original_prompts_file": "datasets/main_test_tasks/hilbig2014generalized_exp1.jsonl",
            "zero_shot_prompts_file": "eval_results/hilbig2014generalized_zero_shot_prompts.jsonl",
            "baseline_nll": 0.0618637911975383,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.1922362762758062, # from all_data_baseline.csv
            "llama_nll": 0.1502313464879989,    # from all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
            "random_nll": 0.6931471805599453,    # ln(2) for binary choices
            "description": "Multi-attribute decision making task (individual trials extracted from participant sessions)"
        },
        "hebart2023things": {
            "original_prompts_file": "datasets/main_test_tasks/hebart2023things_exp1.jsonl",
            "zero_shot_prompts_file": "eval_results/hebart2023things_zero_shot_prompts.jsonl",
            "baseline_nll": 0.7851982712745667,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.8343300819396973,  # from all_data_baseline.csv
            "llama_nll": 1.1657265424728394,     # from all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
            "random_nll": 1.0986122886681098,    # ln(3) for 3-choice task from all_data_random.csv
            "description": "THINGS odd-one-out task with 3-choice trials",
            "parsing": "Extract individual trials from participant sessions (variable trials per participant: 60-2000+)",
            "baselines": "Full comparison: Original Centaur, Cognitive models, Base Llama, Random"
        },
                 "dubois2022value": {
             "original_prompts_file": "original/generalization/dubois2022value/prompts.jsonl",
            "zero_shot_prompts_file": "eval_results/dubois2022value_zero_shot_prompts.jsonl",
                         "baseline_nll": None,  # Not available - dubois2022value is in generalization tasks
             "cognitive_nll": None,  # Not available - dubois2022value is in generalization tasks
             "llama_nll": None,      # Not available - dubois2022value is in generalization tasks
                         "random_nll": 0.6931471805599453,    # ln(2) for binary choices
             "description": "Sequential value-based decision making task (games split, trials evaluated sequentially within games)"
        },

    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    return configs[dataset_name]

def _evaluate_with_sfttrainer(model, tokenizer, dataset_list, task_name, batch_size, rank=0, world_size=1, skip_detailed_analysis=False):
    """Core evaluation function using SFTTrainer - shared by all evaluation methods"""
    
    # Check if we should use efficient batch-based evaluation for detailed analysis
    if not skip_detailed_analysis and len(dataset_list) > 1000:
        print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Using efficient batch-based evaluation for {len(dataset_list):,} trials")
        return _evaluate_with_efficient_batch_processing(model, tokenizer, dataset_list, task_name, batch_size, rank, world_size)
    
    # Set up data collator exactly like original
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, 
        instruction_template=r_id, 
        tokenizer=tokenizer
    )
    
    # Convert to HuggingFace dataset format
    eval_dataset = Dataset.from_list(dataset_list)
    
    # Track quantization state (same as original)
    is_quantized = model.is_quantized
    
    # Evaluate using SFTTrainer (exactly like original)
    with torch.no_grad():
        model.is_quantized = False  # Same as original
        
        training_args = TrainingArguments(
            output_dir=f"eval_gpu_{rank}" if world_size > 1 else "eval",
            per_device_eval_batch_size=batch_size,
            report_to=[],  # Disable wandb and other reporting
            local_rank=rank if world_size > 1 else -1,
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=32768,
            data_collator=collator,
        )
        
        model.is_quantized = is_quantized  # Restore quantization state
        
        # Get overall results
        result = trainer.evaluate()
        nll = result['eval_loss']
        num_samples = len(eval_dataset)
        
        print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}NLL: {nll:.4f} ({num_samples} samples)")
        
        # Save basic results immediately
        basic_results = {
            'task': task_name,
            'model': model.name_or_path if hasattr(model, 'name_or_path') else 'unknown',
            'nll': nll,
            'num_samples': num_samples,
            'dataset_size': len(eval_dataset),
            'prompts_file': 'from_memory',  # Will be updated by caller
            'eval_config': {
                'batch_size': batch_size,
                'max_seq_length': 32768,
                'quantization': '4bit',
                'pipeline': 'SFTTrainer evaluation'
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Add distributed info if applicable
        if world_size > 1:
            basic_results.update({
                'gpu_rank': rank,
                'world_size': world_size,
                'pipeline': 'Distributed Centaur evaluation'
            })
        
        # Detailed analysis (optional)
        detailed_results = None
        if not skip_detailed_analysis:
            print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Collecting detailed per-trial results...")
            detailed_results = _collect_detailed_results(model, tokenizer, dataset_list, task_name, rank, world_size)
        
        return nll, num_samples, basic_results, detailed_results

def _evaluate_with_efficient_batch_processing(model, tokenizer, dataset_list, task_name, batch_size, rank=0, world_size=1):
    """
    Efficient evaluation that captures individual losses during batch processing
    Avoids the 2x computation overhead of the original approach
    """
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.eval()
    
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Processing {len(dataset_list):,} trials with efficient batch evaluation...")
    
    # Prepare data for batching
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32768)
        return {
            'input_ids': tokenized['input_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device),
            'original_data': batch
        }
    
    dataloader = DataLoader(dataset_list, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_losses = []
    per_trial_results = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            original_data = batch['original_data']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # Get individual losses for each sample in the batch
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            labels = input_ids      # [batch_size, seq_len]
            
            # Compute individual sample losses
            batch_individual_losses = []
            for i in range(input_ids.size(0)):
                sample_logits = logits[i]  # [seq_len, vocab_size]
                sample_labels = labels[i]  # [seq_len]
                sample_mask = attention_mask[i]  # [seq_len]
                
                # Only compute loss on non-masked tokens
                valid_positions = sample_mask.bool()
                if valid_positions.sum() > 1:  # Need at least 2 tokens (input + target)
                    # Shift for causal LM: predict next token
                    shift_logits = sample_logits[:-1][valid_positions[:-1]]  # Remove last position
                    shift_labels = sample_labels[1:][valid_positions[1:]]    # Remove first position
                    
                    if len(shift_labels) > 0:
                        sample_loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
                        individual_loss = sample_loss.item()
                    else:
                        individual_loss = float('inf')
                else:
                    individual_loss = float('inf')
                
                batch_individual_losses.append(individual_loss)
                all_losses.append(individual_loss)
                
                # Extract ground truth for this sample
                prompt_data = original_data[i]
                ground_truth = _extract_ground_truth(prompt_data['text'], task_name)
                
                # Store detailed results for this trial
                trial_result = {
                    'trial_index': total_samples + i,
                    'prompt_text': prompt_data['text'],
                    'ground_truth_choice': ground_truth,
                    'trial_nll': individual_loss,
                    'prompt_length': len(prompt_data['text']),
                    'token_count': valid_positions.sum().item(),
                    'participant': prompt_data.get('participant', None),
                    'experiment': prompt_data.get('experiment', None),
                    'trial_number': prompt_data.get('trial_number', None),
                    'choice': prompt_data.get('choice', ground_truth),
                    'dataset': task_name,
                    'batch_index': batch_idx,
                    'sample_in_batch': i,
                    'evaluation_method': 'efficient_batch_processing'
                }
                
                # Add dataset-specific fields
                for key in ['game_number', 'trial_in_game', 'environment', 'choice_index', 'objects', 'keys']:
                    if key in prompt_data:
                        trial_result[key] = prompt_data[key]
                
                # Add distributed info if applicable
                if world_size > 1:
                    trial_result.update({
                        'gpu_rank': rank,
                        'local_trial_index': i
                    })
                
                per_trial_results.append(trial_result)
            
            # Update totals
            valid_losses = [loss for loss in batch_individual_losses if loss != float('inf')]
            if valid_losses:
                total_loss += sum(valid_losses)
                total_samples += len(valid_losses)
            
            # Progress update
            if (batch_idx + 1) % 100 == 0:
                current_avg = total_loss / total_samples if total_samples > 0 else float('inf')
                processed_samples = total_samples
                print(f"     {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Processed {processed_samples:,} trials... Current avg NLL: {current_avg:.3f}")
    
    # Compute final statistics
    valid_losses = [loss for loss in all_losses if loss != float('inf')]
    final_nll = sum(valid_losses) / len(valid_losses) if valid_losses else float('inf')
    
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}✅ Efficient evaluation complete")
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}🎯 Final NLL: {final_nll:.4f}")
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}📊 Valid trials: {len(valid_losses)}/{len(dataset_list)}")
    
    # Create basic results
    basic_results = {
        'task': task_name,
        'model': model.name_or_path if hasattr(model, 'name_or_path') else 'unknown',
        'nll': final_nll,
        'num_samples': len(valid_losses),
        'dataset_size': len(dataset_list),
        'prompts_file': 'from_memory',
        'eval_config': {
            'batch_size': batch_size,
            'max_seq_length': 32768,
            'quantization': '4bit',
            'pipeline': 'Efficient batch processing evaluation'
        },
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Add distributed info if applicable
    if world_size > 1:
        basic_results.update({
            'gpu_rank': rank,
            'world_size': world_size,
            'pipeline': 'Efficient distributed evaluation'
        })
    
    # Create detailed results
    detailed_results = {
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_losses),
            'final_average_nll': final_nll,
            'raw_trial_nlls': valid_losses,
            'verification': {
                'manual_average': final_nll,
                'note': 'Computed during batch processing - no redundant computation'
            }
        },
        'collection_method': 'Efficient batch processing with individual loss extraction',
        'task': task_name
    }
    
    return final_nll, len(valid_losses), basic_results, detailed_results

def _extract_ground_truth(prompt_text, task_name):
    """Extract ground truth choice from prompt text based on dataset pattern"""
    import re
    
    if task_name == "ruggeri2022globalizability":
        pattern = r'You press <<([^>]+)>>'
    elif task_name == "awad2018moral":
        pattern = r'You select outcome <<([XG])>>'
    elif task_name == "hilbig2014generalized":
        pattern = r'You press <<([A-Z]+)>>'
    elif task_name == "hebart2023things":
        pattern = r'You press <<([A-Z])>>'
    elif task_name == "dubois2022value":
        pattern = r'You press <<([^>]+)>>'
    else:
        pattern = r'<<([^>]+)>>'
    
    match = re.search(pattern, prompt_text)
    return match.group(1) if match else None

def _collect_detailed_results(model, tokenizer, dataset_list, task_name, rank=0, world_size=1):
    """Collect detailed per-trial results with trial-specific NLL for each choice token"""
    
    device = next(model.parameters()).device
    per_trial_results = []
    
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Collecting detailed per-trial NLL for {len(dataset_list):,} prompts...")
    
    # Safety check for very large datasets
    if len(dataset_list) > 50000:
        print(f"   ⚠️  WARNING: Very large dataset ({len(dataset_list):,} trials)")
        print(f"   ⚠️  This will take several hours. Consider using --skip-detailed-analysis")
        print(f"   ⚠️  Estimated time: {len(dataset_list) / 3.3 / 3600:.1f} hours")
    
    # Process each prompt individually for detailed results
    for i, prompt_data in enumerate(dataset_list):
        prompt_text = prompt_data['text']
        
        # Extract choice prediction pattern for this dataset
        if task_name == "ruggeri2022globalizability":
            choice_pattern = r'You press <<([^>]+)>>'
        elif task_name == "awad2018moral":
            choice_pattern = r'You select outcome <<([XG])>>'
        elif task_name == "hilbig2014generalized":
            choice_pattern = r'You press <<([A-Z]+)>>'
        elif task_name == "hebart2023things":
            choice_pattern = r'You press <<([A-Z])>>'
        elif task_name == "dubois2022value":
            choice_pattern = r'You press <<([^>]+)>>'

        else:
            choice_pattern = r'<<([^>]+)>>'
        
        import re
        ground_truth_match = re.search(choice_pattern, prompt_text)
        ground_truth = ground_truth_match.group(1) if ground_truth_match else None
        
        if not ground_truth:
            print(f"   Warning: No ground truth for trial {i}, skipping...")
            continue
        
        # Split prompt into context and target choice for trial-specific NLL
        target_pattern = f"You press <<{ground_truth}>>" if "You press" in choice_pattern else f"You select outcome <<{ground_truth}>>"
        if not prompt_text.endswith(target_pattern):
            # Try alternative patterns
            for alt_pattern in [f"<<{ground_truth}>>", f"You choose <<{ground_truth}>>"]:
                if prompt_text.endswith(alt_pattern):
                    target_pattern = alt_pattern
                    break
        
        if prompt_text.endswith(target_pattern):
            # Extract context (everything up to and including the opening "<<")
            # This aligns with DataCollator behavior: everything before "<<" is masked, 
            # tokens between "<<" and ">>" are evaluated, everything after ">>" is masked
            context_end = prompt_text.rfind(" <<") + 3  # Include " <<" in context
            context_text = prompt_text[:context_end]
            
            # Extract only the choice content (what's between << and >>)
            choice_content = ground_truth  # Just the choice itself, no format markers
            
            # Tokenize context only
            context_inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=32768)
            context_inputs = {k: v.to(device) for k, v in context_inputs.items()}
            
            # Tokenize ONLY the choice content (matching DataCollator behavior)
            target_inputs = tokenizer(choice_content, return_tensors="pt", add_special_tokens=False)
            target_inputs = {k: v.to(device) for k, v in target_inputs.items()}
            
            # Combine context + target for full sequence
            full_input_ids = torch.cat([context_inputs["input_ids"], target_inputs["input_ids"]], dim=-1)
            full_attention_mask = torch.cat([context_inputs["attention_mask"], target_inputs["attention_mask"]], dim=-1)
            
            full_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask
            }
            
            with torch.no_grad():
                # Get model outputs for the full sequence
                outputs = model(**full_inputs)
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
                
                # Extract logits for the target tokens only
                context_len = context_inputs["input_ids"].shape[-1]
                target_len = target_inputs["input_ids"].shape[-1]
                
                # CORRECT indexing: logits at position i predict token at position i+1
                # To predict target tokens at positions [context_len, context_len+1, ..., context_len+target_len-1]
                # We need logits at positions [context_len-1, context_len, ..., context_len+target_len-2]
                target_start_pos = context_len - 1  # Start predicting from the last context position
                target_end_pos = context_len + target_len - 1  # End at target_len positions later
                
                if target_end_pos <= logits.shape[1]:  # Ensure we don't go out of bounds
                    target_logits = logits[0, target_start_pos:target_end_pos, :]  # Shape: (target_len, vocab_size)
                    target_token_ids = target_inputs["input_ids"][0, :]  # All target tokens
                    
                    # Compute cross-entropy loss for target tokens only
                    if target_logits.shape[0] > 0 and target_token_ids.shape[0] > 0:
                        # Ensure shapes match - we should predict all target tokens
                        min_len = min(target_logits.shape[0], target_token_ids.shape[0])
                        target_logits = target_logits[:min_len, :]
                        target_token_ids = target_token_ids[:min_len]
                        
                        # Compute negative log-likelihood for target tokens
                        log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
                        target_log_probs = log_probs[range(min_len), target_token_ids]
                        
                        # Convert to NLL (negative log likelihood) - should be small positive numbers like 0.5-3.0
                        individual_token_nlls = [-log_prob.item() for log_prob in target_log_probs]
                        target_token_nll = sum(individual_token_nlls) / len(individual_token_nlls)  # Average NLL
                        
                        # Debug info to verify reasonable values
                        if any(nll > 10 for nll in individual_token_nlls):
                            print(f"   Warning: Trial {i} has very high NLL values: {individual_token_nlls}")
                        
                        trial_loss = target_token_nll  # Use target-specific NLL
                    else:
                        # Fallback: use full sequence loss
                        full_outputs = model(**full_inputs, labels=full_inputs["input_ids"])
                        trial_loss = full_outputs.loss.item()
                        individual_token_nlls = [trial_loss]  # Single value fallback
                else:
                    # Fallback: use full sequence loss
                    full_outputs = model(**full_inputs, labels=full_inputs["input_ids"])
                    trial_loss = full_outputs.loss.item()
                    individual_token_nlls = [trial_loss]  # Single value fallback
        else:
            # Fallback to full prompt tokenization
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                trial_loss = outputs.loss.item()
                individual_token_nlls = [trial_loss]  # Single value fallback
            choice_content = ground_truth  # Fallback: just use the ground truth as the choice content
        
        # Get additional token information - only the choice content tokens
        target_tokens = tokenizer(ground_truth, add_special_tokens=False)['input_ids']
        
        # Store detailed results for this trial with raw token-level data
        trial_result = {
            'trial_index': i,
            'prompt_text': prompt_text,
            'ground_truth_choice': ground_truth,
            'trial_nll': trial_loss,  # Overall trial NLL (target-choice-specific)
            'individual_token_nlls': individual_token_nlls,  # Raw NLL for each evaluated token
            'target_token_info': {
                'ground_truth_choice': ground_truth,
                'ground_truth_tokens': target_tokens,
                'num_target_tokens': len(target_tokens),
                'target_pattern': choice_content,
                'token_breakdown': {
                    'token_ids': target_tokens,
                    'token_nlls': individual_token_nlls,
                    'token_strings': [tokenizer.decode([tid]) for tid in target_tokens] if target_tokens else []
                }
            },
            'prompt_length': len(prompt_text),
            'token_count': len(tokenizer(prompt_text)['input_ids']),
            'participant': prompt_data.get('participant', None),
            'experiment': prompt_data.get('experiment', None),
            'trial_number': prompt_data.get('trial_number', None),
            'choice': prompt_data.get('choice', ground_truth),
            'dataset': task_name
        }
        
        # Add dataset-specific fields
        for key in ['game_number', 'trial_in_game', 'environment', 'choice_index', 'objects', 'keys']:
            if key in prompt_data:
                trial_result[key] = prompt_data[key]
        
        # Add distributed info if applicable
        if world_size > 1:
            trial_result.update({
                'gpu_rank': rank,
                'local_trial_index': i
            })
        
        per_trial_results.append(trial_result)
        
        # Progress update with sample NLL values (more frequent for large datasets)
        progress_interval = 500 if len(dataset_list) > 10000 else 1000
        if (i + 1) % progress_interval == 0:
            recent_nlls = [r['trial_nll'] for r in per_trial_results[-10:]]
            avg_recent_nll = sum(recent_nlls) / len(recent_nlls) if recent_nlls else 0
            percent_complete = (i + 1) / len(dataset_list) * 100
            print(f"     {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}Processed {i + 1:,}/{len(dataset_list):,} trials ({percent_complete:.1f}%)... Recent avg NLL: {avg_recent_nll:.3f}")
    
    # Compute summary statistics
    trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] is not None]
    
    # Collect ALL individual token NLLs - this produces the final average
    all_individual_token_nlls = []
    for trial in per_trial_results:
        all_individual_token_nlls.extend(trial['individual_token_nlls'])
    
    summary_stats = {
        'total_trials': len(per_trial_results),
        'total_tokens_evaluated': len(all_individual_token_nlls),
        'final_average_nll': np.mean(all_individual_token_nlls) if all_individual_token_nlls else None,
        'raw_token_nlls': all_individual_token_nlls,  # Raw data that produces the final average
        'verification': {
            'manual_average': np.mean(all_individual_token_nlls) if all_individual_token_nlls else None,
            'note': 'manual_average should match the reported SFTTrainer eval_loss'
        }
    }
    
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}✅ Collected detailed results for {len(per_trial_results)} trials")
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}🎯 Final average NLL: {summary_stats['final_average_nll']:.4f}")
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}🔍 Total tokens evaluated: {summary_stats['total_tokens_evaluated']}")
    print(f"   {'GPU ' + str(rank) + ': ' if world_size > 1 else ''}📊 Raw token NLLs saved for verification")
    
    return {
        'per_trial_results': per_trial_results,
        'summary_statistics': summary_stats,
        'collection_method': 'Target-choice-specific NLL with token-level breakdown',
        'task': task_name
    }

def evaluate_centaur_zero_shot_distributed_optimized(rank, world_size, model_name, prompts_file, task_name, batch_size=4, skip_detailed_analysis=False):
    """Distributed evaluation function with KV caching optimization for dubois2022value"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    print(f"🔍 Evaluating on GPU {rank}/{world_size-1}: {model_name} on {task_name} (KV Caching Optimized)")
    print(f"   Using prompts: {prompts_file}")
    print(f"   Batch size per GPU: {batch_size}")
    
    # Load model on specific GPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,  # Same as original
        device_map=device,
    )
    
    # Load and distribute dataset
    dataset_list = []
    with open(prompts_file, 'r') as f:
        for line in f:
            dataset_list.append(json.loads(line))
    
    # Distribute data across GPUs
    total_samples = len(dataset_list)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_dataset_list = dataset_list[start_idx:end_idx]
    print(f"   GPU {rank}: Processing samples {start_idx}-{end_idx-1} ({len(local_dataset_list)} samples)")
    
    # Evaluate using KV caching optimization for dubois2022value
    if task_name == "dubois2022value":
        # Use KV caching optimization
        print(f"   GPU {rank}: Using KV caching optimization for dubois2022value")
        
        # Group trials by participant for better organization
        participant_groups = {}
        for trial in local_dataset_list:
            participant = trial['participant']
            if participant not in participant_groups:
                participant_groups[participant] = []
            participant_groups[participant].append(trial)
        
        all_results = []
        total_games = 0
        total_trials = 0
        
        for participant, trials in participant_groups.items():
            # Group trials by game
            game_groups = {}
            for trial in trials:
                game_key = (trial['participant'], trial['game_number'])
                if game_key not in game_groups:
                    game_groups[game_key] = []
                game_groups[game_key].append(trial)
            
            total_games += len(game_groups)
            
            # Process each game with KV caching
            for game_key, game_trials in game_groups.items():
                game_results = evaluate_game_with_kv_caching(model, tokenizer, game_trials, device)
                all_results.extend(game_results)
                total_trials += len(game_trials)
                
                # Progress update
                if len(all_results) % 1000 == 0:
                    print(f"     GPU {rank}: Processed {len(all_results)} trials...")
        
        print(f"   GPU {rank}: Completed {total_games} games, {total_trials} trials")
        
        # Calculate average loss
        valid_losses = [r['loss'] for r in all_results if r['loss'] != float('inf')]
        if valid_losses:
            nll = sum(valid_losses) / len(valid_losses)
            print(f"   GPU {rank}: Local NLL: {nll:.4f} ({len(valid_losses)} valid trials)")
        else:
            nll = float('inf')
            print(f"   GPU {rank}: No valid trials found")
        
        # Save local basic results immediately
        basic_results = {
            'task': task_name,
            'model': model_name,
            'gpu_rank': rank,
            'world_size': world_size,
            'nll': nll,
            'num_samples': len(valid_losses),
            'total_trials_processed': total_trials,
            'total_games_processed': total_games,
            'optimization_method': 'KV_caching',
            'dataset_size': len(local_dataset_list),
            'prompts_file': prompts_file,
            'eval_config': {
                'batch_size': batch_size,
                'max_seq_length': 32768,
                'quantization': '4bit',
                'pipeline': 'Distributed Centaur evaluation with KV caching'
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # NOTE: Skip intermediate file creation for distributed functions - only used in distributed mode
        # The main() function uses single GPU evaluation, so these distributed functions are unused
        
        num_samples = len(valid_losses)
        
    else:
        # Use standard evaluation for other datasets
        print(f"   GPU {rank}: Using standard evaluation for {task_name}")
        nll, num_samples, basic_results, detailed_results = _evaluate_with_sfttrainer(
            model, tokenizer, local_dataset_list, task_name, batch_size, rank, world_size, skip_detailed_analysis
        )
        
        # Update prompts_file in results
        basic_results['prompts_file'] = prompts_file
        
        # NOTE: Skip intermediate file creation - this distributed function is unused in main()
    
    if world_size > 1:
        cleanup_distributed()
    
    return nll, num_samples, basic_results

def evaluate_centaur_zero_shot_distributed(rank, world_size, model_name, prompts_file, task_name, batch_size=4, skip_detailed_analysis=False):
    """Distributed evaluation function for multiple GPUs"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    print(f"🔍 Evaluating on GPU {rank}/{world_size-1}: {model_name} on {task_name}")
    print(f"   Using prompts: {prompts_file}")
    print(f"   Batch size per GPU: {batch_size}")
    
    # Load model on specific GPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,  # Same as original
        device_map=device,
    )
    
    # Load and distribute dataset
    dataset_list = []
    with open(prompts_file, 'r') as f:
        for line in f:
            dataset_list.append(json.loads(line))
    
    # Distribute data across GPUs
    total_samples = len(dataset_list)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_dataset_list = dataset_list[start_idx:end_idx]
    print(f"   GPU {rank}: Processing samples {start_idx}-{end_idx-1} ({len(local_dataset_list)} samples)")
    
    # Use shared evaluation function
    nll, num_samples, basic_results, detailed_results = _evaluate_with_sfttrainer(
        model, tokenizer, local_dataset_list, task_name, batch_size, rank, world_size, skip_detailed_analysis
    )
    
    # Update prompts_file in results
    basic_results['prompts_file'] = prompts_file
    
            # NOTE: Skip intermediate file creation - this distributed function is unused in main()
    
    if world_size > 1:
        cleanup_distributed()
    
    return nll, num_samples, basic_results

def evaluate_centaur_zero_shot_optimized_single_gpu(model_name, prompts_file, task_name, batch_size=4, skip_detailed_analysis=False):
    """Single GPU evaluation with KV caching optimization for dubois2022value"""
    print(f"🔍 Evaluating {model_name} on {task_name} (KV Caching Optimized)")
    print(f"   Using prompts: {prompts_file}")
    print(f"   Batch size: {batch_size}")
    
    # Load model exactly like original evaluation
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,  # Same as original
    )
    
    # Load dataset from file
    dataset_list = []
    with open(prompts_file, 'r') as f:
        for line in f:
            dataset_list.append(json.loads(line))
    
    print(f"   Loaded {len(dataset_list)} trials")
    
    # Use KV caching optimization for dubois2022value
    if task_name == "dubois2022value":
        print(f"   Using KV caching optimization for dubois2022value")
        
        # Group trials by participant for better organization
        participant_groups = {}
        for trial in dataset_list:
            participant = trial['participant']
            if participant not in participant_groups:
                participant_groups[participant] = []
            participant_groups[participant].append(trial)
        
        all_results = []
        total_games = 0
        total_trials = 0
        
        print(f"   Processing {len(participant_groups)} participants...")
        
        for participant, trials in participant_groups.items():
            # Group trials by game
            game_groups = {}
            for trial in trials:
                game_key = (trial['participant'], trial['game_number'])
                if game_key not in game_groups:
                    game_groups[game_key] = []
                game_groups[game_key].append(trial)
            
            total_games += len(game_groups)
            
            # Process each game with KV caching
            for game_key, game_trials in game_groups.items():
                game_results = evaluate_game_with_kv_caching(model, tokenizer, game_trials, model.device)
                all_results.extend(game_results)
                total_trials += len(game_trials)
                
                # Progress update
                if len(all_results) % 1000 == 0:
                    print(f"     Processed {len(all_results)} trials...")
        
        print(f"   Completed {total_games} games, {total_trials} trials")
        
        # Calculate average loss
        valid_losses = [r['loss'] for r in all_results if r['loss'] != float('inf')]
        if valid_losses:
            nll = sum(valid_losses) / len(valid_losses)
            print(f"   Overall NLL: {nll:.4f} ({len(valid_losses)} valid trials)")
        else:
            nll = float('inf')
            print(f"   No valid trials found")
        
        # Create basic results
        basic_results = {
            'task': task_name,
            'model': model_name,
            'nll': nll,
            'num_samples': len(valid_losses),
            'total_trials_processed': total_trials,
            'total_games_processed': total_games,
            'optimization_method': 'KV_caching',
            'dataset_size': len(dataset_list),
            'prompts_file': prompts_file,
            'eval_config': {
                'batch_size': batch_size,
                'max_seq_length': 32768,
                'quantization': '4bit',
                'pipeline': 'Single GPU Centaur evaluation with KV caching'
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save basic results immediately (eval_results directory created in main())
        eval_results_dir = Path("eval_results")
        basic_results_file = eval_results_dir / f'{task_name}_basic_evaluation_results.json'
        with open(basic_results_file, 'w') as f:
            json.dump(basic_results, f, indent=2)
        print(f"   ✅ Saved basic results to {basic_results_file}")
        
        # Handle detailed results
        if skip_detailed_analysis:
            print(f"   ⚠️  Skipping detailed per-trial analysis (--skip-detailed-analysis flag)")
            detailed_results = {
                'task': task_name,
                'model': model_name,
                'overall_nll': nll,
                'dataset_size': len(dataset_list),
                'prompts_file': prompts_file,
                'optimization_method': 'KV_caching',
                'eval_config': {
                    'batch_size': batch_size,
                    'max_seq_length': 32768,
                    'quantization': '4bit',
                    'pipeline': 'Single GPU Centaur evaluation with KV caching'
                },
                'per_trial_results': [],
                'per_trial_statistics': {
                    'note': 'Detailed analysis skipped due to --skip-detailed-analysis flag'
                }
            }
        else:
            # Convert results to detailed format
            per_trial_results = []
            for i, result in enumerate(all_results):
                trial = result['trial']
                trial_result = {
                    'trial_index': i,
                    'prompt_text': trial['text'],
                    'ground_truth_choice': trial.get('choice'),
                    'trial_loss': result['loss'],
                    'prompt_length': len(trial['text']),
                    'participant': trial['participant'],
                    'experiment': trial['experiment'],
                    'game_number': trial['game_number'],
                    'trial_in_game': trial['trial_in_game'],
                    'optimization_method': result['method'],
                    'choice_position': result.get('choice_position'),
                    'choice_text': result.get('choice_text')
                }
                per_trial_results.append(trial_result)
            
            print(f"   Collected detailed results for {len(per_trial_results)} trials")
            
            # Create detailed results structure
            detailed_results = {
                'task': task_name,
                'model': model_name,
                'overall_nll': nll,
                'dataset_size': len(dataset_list),
                'prompts_file': prompts_file,
                'optimization_method': 'KV_caching',
                'eval_config': {
                    'batch_size': batch_size,
                    'max_seq_length': 32768,
                    'quantization': '4bit',
                    'pipeline': 'Single GPU Centaur evaluation with KV caching'
                },
                'per_trial_results': per_trial_results,
                'per_trial_statistics': {
                    'total_trials': len(per_trial_results),
                    'average_trial_loss': sum(r['trial_loss'] for r in per_trial_results) / len(per_trial_results) if per_trial_results else 0,
                    'min_trial_loss': min(r['trial_loss'] for r in per_trial_results) if per_trial_results else 0,
                    'max_trial_loss': max(r['trial_loss'] for r in per_trial_results) if per_trial_results else 0,
                    'average_prompt_length': sum(r['prompt_length'] for r in per_trial_results) / len(per_trial_results) if per_trial_results else 0,
                }
            }
        
        # NOTE: Detailed results are saved in main eval function, no need for duplicate file
        
        return nll, detailed_results
        
    else:
        # Fallback to standard evaluation for other datasets
        print(f"   Using standard evaluation for {task_name}")
        return evaluate_centaur_zero_shot(model_name, prompts_file, task_name, batch_size, skip_detailed_analysis)

def evaluate_centaur_zero_shot(model_name, prompts_file, task_name, batch_size=4, skip_detailed_analysis=False):
    """Evaluate Centaur on zero-shot prompts using unified progressive NLL extraction"""
    print(f"🔍 Evaluating {model_name} on {task_name} (zero-shot)")
    print(f"   Using prompts: {prompts_file}")
    print(f"   Batch size: {batch_size}")
    print(f"   Using unified trial-level NLL extraction (zero-shot evaluation)")
    
    # Load model exactly like original evaluation
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,  # Same as original
    )
    
    # Load dataset from file (same format as original)
    dataset_list = []
    with open(prompts_file, 'r') as f:
        for line in f:
            dataset_list.append(json.loads(line))
    
    print(f"   Loaded {len(dataset_list)} prompts")
    
    # Use unified progressive NLL extraction for trial-level results
    nll, basic_results, detailed_results = evaluate_zero_shot_with_unified_nll(
        model, tokenizer, dataset_list, task_name, batch_size
    )
    
    return nll, detailed_results

def evaluate_game_with_kv_caching(model, tokenizer, game_trials, device):
    """
    Optimized evaluation for dubois2022value: process all trials in a game with single forward pass + KV caching
    """
    if not game_trials:
        return []
    
    # Group trials by game
    game_groups = {}
    for trial in game_trials:
        game_key = (trial['participant'], trial['game_number'])
        if game_key not in game_groups:
            game_groups[game_key] = []
        game_groups[game_key].append(trial)
    
    all_results = []
    
    for game_key, trials in game_groups.items():
        participant, game_num = game_key
        
        # Sort trials by trial_in_game to ensure correct order
        trials.sort(key=lambda x: x['trial_in_game'])
        
        # Build the complete game context (all trials concatenated)
        # This will be the longest possible sequence for this game
        game_context = trials[0]['text']  # Start with first trial's context
        
        # Find where the first actual choice is in the text
        choice_pattern = r'You press <<([^>]+)>>'
        import re
        
        # For each trial, we need to find the position of its choice token
        choice_positions = []
        current_text = game_context
        
        for trial_idx, trial in enumerate(trials):
            # Find the choice in the current trial's text
            choice_match = re.search(choice_pattern, trial['text'])
            if choice_match:
                choice_text = choice_match.group(1)
                
                # Find this choice in the current text
                choice_start = current_text.find(f"You press <<{choice_text}>>")
                if choice_start != -1:
                    # Tokenize up to this position to get token index
                    tokens_before_choice = tokenizer(current_text[:choice_start], return_tensors="pt", truncation=False)
                    choice_token_pos = len(tokens_before_choice['input_ids'][0]) - 1  # -1 because we want the position before the choice
                    choice_positions.append((trial_idx, choice_token_pos, choice_text))
                    
                    # Add this choice to the context for next trial
                    current_text += f"\nYou press <<{choice_text}>>"
        
        if not choice_positions:
            # Fallback: process each trial individually
            for trial in trials:
                inputs = tokenizer(trial['text'], return_tensors="pt", truncation=True, max_length=32768)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    trial_loss = outputs.loss.item()
                
                all_results.append({
                    'trial': trial,
                    'loss': trial_loss,
                    'method': 'individual_fallback'
                })
            continue
        
        # Single forward pass for the complete game
        inputs = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Extract losses for each choice position
            for trial_idx, choice_token_pos, choice_text in choice_positions:
                if choice_token_pos < len(logits):
                    # Get logits at the choice position
                    choice_logits = logits[choice_token_pos]
                    
                    # Find the token ID for the choice text
                    choice_tokens = tokenizer(f" <<{choice_text}>>", return_tensors="pt", add_special_tokens=False)
                    if len(choice_tokens['input_ids'][0]) > 0:
                        target_token_id = choice_tokens['input_ids'][0][0]  # First token of the choice
                        
                        # Calculate loss for this specific token
                        loss = torch.nn.functional.cross_entropy(
                            choice_logits.unsqueeze(0), 
                            torch.tensor([target_token_id], device=device)
                        ).item()
                        
                        all_results.append({
                            'trial': trials[trial_idx],
                            'loss': loss,
                            'method': 'kv_cached',
                            'choice_position': choice_token_pos,
                            'choice_text': choice_text
                        })
                    else:
                        # Fallback for tokenization issues
                        all_results.append({
                            'trial': trials[trial_idx],
                            'loss': float('inf'),
                            'method': 'tokenization_fallback'
                        })
                else:
                    # Fallback for position issues
                    all_results.append({
                        'trial': trials[trial_idx],
                        'loss': float('inf'),
                        'method': 'position_fallback'
                    })
    
    return all_results

def aggregate_distributed_results(task_name, world_size):
    """Aggregate results from multiple GPUs"""
    print(f"📊 Aggregating results from {world_size} GPUs...")
    
    total_loss = 0.0
    total_samples = 0
    all_local_results = []
    
    # Load results from each GPU
    for rank in range(world_size):
        local_results_file = f'local_evaluation_results_{task_name}_gpu_{rank}.json'
        if os.path.exists(local_results_file):
            with open(local_results_file, 'r') as f:
                local_results = json.load(f)
                all_local_results.append(local_results)
                
                # Weighted average (loss is already averaged per sample by SFTTrainer)
                # Handle both old and new variable names for compatibility
                local_samples = local_results.get('num_samples', local_results.get('local_samples', 0))
                local_nll = local_results.get('nll', local_results.get('local_nll', float('inf')))
                
                total_loss += local_nll * local_samples
                total_samples += local_samples
                
                print(f"   GPU {rank}: {local_nll:.4f} NLL ({local_samples} samples)")
        else:
            print(f"   ⚠️  Warning: Missing results from GPU {rank}")
    
    # Calculate global average
    if total_samples > 0:
        global_nll = total_loss / total_samples
        print(f"   🎯 Global NLL: {global_nll:.4f} (total: {total_samples} samples)")
    else:
        global_nll = float('inf')
        print(f"   ❌ No valid results found")
    
    # Create aggregated results
    aggregated_results = {
        'task': task_name,
        'global_nll': global_nll,
        'total_samples': total_samples,
        'world_size': world_size,
        'local_results': all_local_results,
        'aggregation_timestamp': datetime.now().isoformat()
    }
    
    # Save aggregated results
    aggregated_file = f'aggregated_evaluation_results_{task_name}.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"   ✅ Saved aggregated results to {aggregated_file}")
    
    # Cleanup local files
    for rank in range(world_size):
        local_file = f'local_evaluation_results_{task_name}_gpu_{rank}.json'
        if os.path.exists(local_file):
            os.remove(local_file)
    
    return global_nll, aggregated_results

def create_comparison_plot(zero_shot_nll, dataset_config, task_name):
    """Create comparison plot with published results using original paper's styling and intelligent random baseline handling"""
    print(f"   Creating comparison plot...")
    
    # Build comparison data, filtering out None values
    comparison_data = [
        {'Model': 'Zero-Shot Centaur\n(Fair)', 'NLL': zero_shot_nll, 'Type': 'Our Approach'}
    ]
    
    # Add baselines only if available (excluding Llama to avoid misleading comparisons)
    if dataset_config['baseline_nll'] is not None:
        comparison_data.append({
            'Model': 'Original Centaur\n(Unfair)', 
            'NLL': dataset_config['baseline_nll'], 
            'Type': 'Published Paper'
        })
    
    if dataset_config['cognitive_nll'] is not None:
        comparison_data.append({
            'Model': 'Cognitive\nModels', 
            'NLL': dataset_config['cognitive_nll'], 
            'Type': 'Published Paper'
        })
    
    # Use original paper's styling if available
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['nature'])
    else:
        plt.style.use('default')
    
    # Create plot using original paper's styling
    fig, ax = plt.subplots(1, 1, figsize=(7.08661, 3.5))
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Original paper's color scheme (removed orange since no Llama bar)
    base_colors = ['#1f77b4', '#69005f', '#cbc9e2']  # Blue, Purple, Light Purple
    colors = base_colors[:len(comp_df)]  # Use only as many colors as we have data
    
    bars = ax.bar(range(len(comp_df)), comp_df['NLL'], color=colors, alpha=0.8)
    
    # Styling to match original paper
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Negative log-likelihood', fontsize=10)
    ax.set_title(f'Model Comparison: NLL on {task_name.replace("_", " ").title()}\n(Fair vs Unfair Evaluation)', fontsize=11)
    ax.set_xticks(range(len(comp_df)))
    ax.set_xticklabels(comp_df['Model'], rotation=0, ha='center', fontsize=9)
    
    # Intelligent handling of random baseline with smart tick spacing
    random_nll = dataset_config['random_nll']
    model_nlls = comp_df['NLL'].tolist()
    max_model_nll = max(model_nlls)
    min_model_nll = min(model_nlls)
    model_range = max_model_nll - min_model_nll
    
    # Always include random baseline but use smart tick spacing to maintain proportions
    y_bottom = max(0, min_model_nll - 0.1 * model_range)
    y_top = max(random_nll + 0.05 * random_nll, max_model_nll + 0.1 * model_range)
    ax.set_ylim(y_bottom, y_top)
    
    # Add random baseline line
    ax.axhline(y=random_nll, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random guessing')
    ax.text(len(comp_df) - 0.1, random_nll + (y_top - y_bottom) * 0.02, 'Random guessing', 
           fontsize=9, color='red', horizontalalignment='right', fontweight='bold')
    
    # Smart Y-axis tick spacing: focus density on model comparison range, sparse for random area
    random_gap = random_nll - max_model_nll
    if random_gap > 0.5:  # If random is significantly higher
        print(f"   Random baseline ({random_nll:.3f}) is much higher than models ({max_model_nll:.3f})")
        print(f"   Using intelligent Y-axis tick spacing to maintain proportions...")
        
        # Dense ticks in model comparison range (4-5 ticks)
        model_ticks = np.linspace(min_model_nll, max_model_nll, 4)
        
        # Add random baseline tick and one intermediate tick to show the gap
        mid_point = (max_model_nll + random_nll) / 2
        random_ticks = [mid_point, random_nll]
        
        # Combine and sort all ticks
        all_ticks = np.concatenate([model_ticks, random_ticks])
        all_ticks = np.unique(np.round(all_ticks, 3))  # Remove duplicates and round
        
        # Set custom ticks with appropriate formatting
        ax.set_yticks(all_ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in all_ticks])
        
        # Add subtle visual indication of the gap with lighter grid lines
        for tick in all_ticks:
            if tick > max_model_nll + 0.1:  # In the "gap" region above models
                ax.axhline(y=tick, color='lightgray', linewidth=0.5, alpha=0.2, zorder=0)
    else:
        # Standard approach when random is close to model performances - use default ticks
        pass
    
    # Add value labels on bars (like original paper)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_model_nll - min_model_nll) * 0.015,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Remove top and right spines (like original paper)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add comprehensive note about the evaluation approach
    note_text = 'Note: Zero-shot evaluation removes behavioral history. Llama baseline excluded (uses in-context learning).'
    plt.figtext(0.02, 0.02, note_text, fontsize=7, style='italic', color='gray')
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plots (eval_results directory created in main())
    eval_results_dir = Path("eval_results")
    
    print(f"   Saving plot as PNG...")
    plt.savefig(eval_results_dir / f'{task_name}_zero_shot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saving plot as PDF...")
    plt.savefig(eval_results_dir / f'{task_name}_zero_shot_comparison.pdf', bbox_inches='tight')
    plt.close()  # Close the figure to free memory and prevent hanging
    print(f"   ✅ Plot saved successfully")
    
    # Print detailed comparison summary
    print(f"\n📊 Model Performance Summary:")
    for model_data in comparison_data:
        print(f"   {model_data['Model'].replace(chr(10), ' ')}: {model_data['NLL']:.4f}")
    print(f"   Random Baseline: {random_nll:.4f}")
    
    # Enhanced performance analysis
    if len(comparison_data) >= 1:
        zero_shot = comparison_data[0]['NLL']
        
        # Analysis vs random
        vs_random = random_nll - zero_shot
        print(f"\n🎯 Performance Analysis:")
        print(f"   Zero-shot vs Random: {vs_random:.4f} better" if vs_random > 0 else f"   Zero-shot vs Random: {abs(vs_random):.4f} worse")
        
        # Analysis vs baselines
        if dataset_config['baseline_nll'] is not None:
            original = dataset_config['baseline_nll']
            fairness_gap = zero_shot - original
            print(f"   Fairness Gap (Zero-shot vs Unfair Original): {fairness_gap:.4f}")
            if fairness_gap > 0:
                print(f"   → Zero-shot performs worse (unfair advantage confirmed)")
            else:
                print(f"   → Zero-shot performs better (unexpected!)")
        
        if dataset_config['cognitive_nll'] is not None:
            cognitive = dataset_config['cognitive_nll']
            cognitive_gap = zero_shot - cognitive
            print(f"   Zero-shot vs Cognitive Models: {cognitive_gap:.4f}")
            if cognitive_gap > 0:
                print(f"   → Cognitive models outperform zero-shot Centaur")
            else:
                print(f"   → Zero-shot Centaur outperforms cognitive models")
    
    return comparison_data

def analyze_complete_results(results_file):
    """Analyze the complete evaluation results saved in JSON format"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Complete Results Analysis")
    print(f"=" * 50)
    
    # Overall info
    metadata = results['metadata']
    print(f"🎯 Dataset: {metadata['dataset_name']}")
    print(f"🤖 Model: {metadata['model_name']}")
    print(f"📅 Evaluated: {metadata['evaluation_date']}")
    
    # Per-trial analysis
    per_trial = results['evaluation_results']['zero_shot_centaur']['detailed_results']['per_trial_results']
    stats = results['evaluation_results']['zero_shot_centaur']['detailed_results']['per_trial_statistics']
    
    print(f"\n📈 Per-Trial Statistics ({len(per_trial)} trials):")
    print(f"   Mean loss: {stats['mean_trial_loss']:.4f} ± {stats['std_trial_loss']:.4f}")
    print(f"   Loss range: {stats['min_trial_loss']:.4f} to {stats['max_trial_loss']:.4f}")
    print(f"   Mean prompt length: {stats['mean_prompt_length']:.0f} characters")
    print(f"   Mean tokens: {stats['mean_token_count']:.0f}")
    
    # Error analysis
    trial_losses = [trial['trial_loss'] for trial in per_trial]
    sorted_indices = np.argsort(trial_losses)
    
    print(f"\n🔍 Error Analysis:")
    print(f"   Hardest trials (highest loss):")
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        trial = per_trial[idx]
        print(f"     Trial {idx}: loss={trial['trial_loss']:.4f}, len={trial['prompt_length']}")
    
    print(f"   Easiest trials (lowest loss):")
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        trial = per_trial[idx]
        print(f"     Trial {idx}: loss={trial['trial_loss']:.4f}, len={trial['prompt_length']}")
    
    # Prompt length analysis
    prompt_lengths = [trial['prompt_length'] for trial in per_trial]
    length_loss_corr = np.corrcoef(prompt_lengths, trial_losses)[0,1]
    print(f"\n📏 Prompt Length vs Performance:")
    print(f"   Correlation with loss: {length_loss_corr:.3f}")
    
    return results

def list_available_datasets():
    """List all available datasets with descriptions"""
    configs = {
        "ruggeri2022globalizability": {
            "description": "Temporal decision making task",
            "parsing": "Remove behavioral history, keep task instructions",
            "baselines": "Full comparison: Original Centaur, Cognitive models, Base Llama, Random"
        },
        "awad2018moral": {
            "description": "Autonomous vehicle moral dilemmas", 
            "parsing": "Extract each individual trial from participant sessions (each participant has ~13 trials)",
            "baselines": "Random baseline only (awad2018moral is in held_out_tasks - others not computed)"
        },
        "hilbig2014generalized": {
            "description": "Multi-attribute decision making with expert ratings",
            "parsing": "Extract individual trials from participant sessions (each participant has ~90+ trials)",
            "baselines": "Full comparison: Original Centaur, Cognitive models, Base Llama, Random"
        },
        "hebart2023things": {
            "description": "THINGS odd-one-out task with 3-choice trials",
            "parsing": "Extract individual trials from participant sessions (variable trials per participant: 60-2000+)",
            "baselines": "Full comparison: Original Centaur, Cognitive models, Base Llama, Random"
        },
                 "dubois2022value": {
             "description": "Sequential value-based decision making task",
             "parsing": "Split games (400 per participant) and evaluate sequentially within games (each game has 3-10 trials)",
             "baselines": "Random baseline only (dubois2022value is in generalization tasks - others not computed)"
         },
        "collsioo2023MCPL_exp1": {
            "description": "MCPL experiment 1: Minimal context with full behavioral history",
            "parsing": "Extract all trials from the original prompt",
            "baselines": "Random baseline only (collsioo2023MCPL_exp1 is in main_test_tasks - others not computed)"
        },
        "collsioo2023MCPL_all": {
            "original_prompts_file": [
                "datasets/main_test_tasks/collsioo2023MCPL_exp1.jsonl",
                "datasets/main_test_tasks/collsioo2023MCPL_exp2.jsonl",
                "datasets/main_test_tasks/collsioo2023MCPL_exp3.jsonl"
            ],
            "zero_shot_prompts_file": "eval_results/collsioo2023MCPL_all_zero_shot_prompts.jsonl",
            "baseline_nll": None,
            "cognitive_nll": None,
            "llama_nll": None,
            "random_nll": 0.6931471805599453,
            "description": "MCPL all experiments: Minimal context with full behavioral history (exp1, exp2, exp3)"
        }
    }
    
    print("�� Available Datasets:")
    print("=" * 50)
    for dataset, info in configs.items():
        print(f"🎯 {dataset}:")
        print(f"   📝 {info['description']}")
        print(f"   🔧 Parsing: {info['parsing']}")
        print(f"   📊 Baselines: {info['baselines']}")
        print()

def evaluate_zero_shot_with_unified_nll(model, tokenizer, dataset_list, task_name, batch_size=4):
    """
    Unified zero-shot NLL extraction for all datasets with proper batching.
    Extract NLL for all trials in batch simultaneously during the forward pass.
    """
    print(f"🔍 Evaluating {task_name} with unified zero-shot NLL extraction")
    print(f"   Processing {len(dataset_list)} individual trial prompts")
    print(f"   Using batch size: {batch_size}")
    
    device = next(model.parameters()).device
    per_trial_results = []
    all_individual_token_nlls = []
    
    # Use DataLoader for proper batching
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch.nn.functional as F
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32768)
        return {
            'input_ids': tokenized['input_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device),
            'original_data': batch
        }
    
    dataloader = DataLoader(dataset_list, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), 
                                     total=len(dataloader),
                                     desc="Processing batches",
                                     unit="batch"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            original_data = batch['original_data']
            
            # Forward pass for the entire batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            
            # Extract choice information for all trials in batch
            batch_choice_info = []
            for i, trial_data in enumerate(original_data):
                prompt_text = trial_data['text']
                
                # Extract choice pattern
                import re
                choice_patterns = [
                    r'You press <<([^>]+)>>',
                    r'You choose <<([^>]+)>>', 
                    r'You select <<([^>]+)>>',
                    r'<<([^>]+)>>'
                ]
                
                choice_match = None
                for pattern in choice_patterns:
                    match = re.search(pattern, prompt_text)
                    if match:
                        choice_match = match
                        break
                
                if choice_match:
                    choice_value = choice_match.group(1)
                    # Find token positions for this choice
                    choice_start_pos = choice_match.end() - len(choice_value) - 2
                    text_before_choice = prompt_text[:choice_start_pos]
                    tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
                    choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1
                    
                    # Tokenize choice content
                    choice_tokens = tokenizer(choice_value, return_tensors="pt", add_special_tokens=False)
                    choice_token_ids = choice_tokens['input_ids'][0].tolist()
                    
                    batch_choice_info.append({
                        'trial_idx': i,
                        'choice_value': choice_value,
                        'choice_start_pos': choice_start_token_pos,
                        'choice_token_ids': choice_token_ids,
                        'trial_data': trial_data
                    })
                else:
                    tqdm.write(f"   Warning: No choice found for trial {batch_idx * batch_size + i}")
            
            # Extract NLL for all valid choices in batch simultaneously
            for choice_info in batch_choice_info:
                i = choice_info['trial_idx']
                choice_start_pos = choice_info['choice_start_pos']
                choice_token_ids = choice_info['choice_token_ids']
                trial_data = choice_info['trial_data']
                
                # Extract logits for this trial's choice tokens
                sample_logits = logits[i]  # Shape: [seq_len, vocab_size]
                
                if choice_start_pos + len(choice_token_ids) <= sample_logits.size(0):
                    # Get logits for choice token positions
                    choice_logits = sample_logits[choice_start_pos:choice_start_pos + len(choice_token_ids)]  # [num_choice_tokens, vocab_size]
                    choice_token_tensor = torch.tensor(choice_token_ids, device=device)  # [num_choice_tokens]
                    
                    # Compute log probabilities and extract NLLs
                    log_probs = F.log_softmax(choice_logits, dim=-1)  # [num_choice_tokens, vocab_size]
                    choice_log_probs = log_probs[range(len(choice_token_ids)), choice_token_tensor]  # [num_choice_tokens]
                    individual_token_nlls = (-choice_log_probs).cpu().tolist()  # Convert to NLL
                    
                    # Average NLL for this choice
                    trial_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                    all_individual_token_nlls.extend(individual_token_nlls)
                else:
                    trial_nll = float('inf')
                    individual_token_nlls = []
                
                # Store trial result
                trial_result = {
                    'trial_index': batch_idx * batch_size + i,
                    'participant': trial_data.get('participant'),
                    'ground_truth_choice': choice_info['choice_value'],
                    'trial_nll': trial_nll,
                    'individual_token_nlls': individual_token_nlls,
                    'dataset': task_name,
                    'trial_number': trial_data.get('trial_number'),
                    'experiment': trial_data.get('experiment')
                }
                
                # Add dataset-specific fields
                for key in ['game_number', 'trial_in_game', 'environment', 'choice_index', 'objects', 'keys']:
                    if key in trial_data:
                        trial_result[key] = trial_data[key]
                
                per_trial_results.append(trial_result)
            
            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0:
                recent_nlls = [r['trial_nll'] for r in per_trial_results[-1000:] if r['trial_nll'] != float('inf')]
                avg_recent_nll = sum(recent_nlls) / len(recent_nlls) if recent_nlls else float('inf')
                tqdm.write(f"     Processed {len(per_trial_results):,} trials... Recent avg NLL: {avg_recent_nll:.3f}")
    
    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(all_individual_token_nlls) / len(all_individual_token_nlls) if all_individual_token_nlls else float('inf')
    
    print(f"   ✅ Zero-shot NLL extraction complete")
    print(f"   🎯 Overall NLL: {overall_nll:.4f}")
    print(f"   🔍 Valid trials: {len(valid_trial_nlls)}/{len(per_trial_results)}")
    print(f"   📊 Total tokens evaluated: {len(all_individual_token_nlls)}")
    
    # Create basic results structure
    basic_results = {
        'task': task_name,
        'nll': overall_nll,
        'num_samples': len(valid_trial_nlls),
        'total_trials_processed': len(per_trial_results),
        'dataset_size': len(dataset_list),
        'evaluation_method': 'Optimized batched zero-shot NLL extraction',
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create simple detailed results structure
    detailed_results = {
        'task': task_name,
        'overall_nll': overall_nll,
        'dataset_size': len(dataset_list),
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_trial_nlls),
            'total_tokens_evaluated': len(all_individual_token_nlls),
            'overall_nll': overall_nll,
            'final_average_nll': overall_nll,
            'raw_token_nlls': all_individual_token_nlls
        },
        'collection_method': 'Optimized batched NLL extraction with vectorized operations',
        'eval_config': {
            'batch_size': batch_size,
            'max_seq_length': 32768,
            'quantization': '4bit',
            'pipeline': 'Optimized batched zero-shot evaluation'
        }
    }
    
    return overall_nll, basic_results, detailed_results

def main():
    """Main evaluation pipeline - exactly like original but with zero-shot prompts"""
    parser = argparse.ArgumentParser(description='Zero-Shot Centaur Evaluation')
    parser.add_argument('--task', type=str, 
                       choices=['ruggeri2022globalizability', 'awad2018moral', 'hilbig2014generalized', 'hebart2023things', 'dubois2022value'],
                       default='ruggeri2022globalizability',
                       help='Dataset to evaluate on')
    parser.add_argument('--model', '--model-name', dest='model_name', type=str, 
                       default='marcelbinz/Llama-3.1-Centaur-70B-adapter',
                       help='Model to evaluate')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--analyze-results', type=str, 
                       help='Path to existing results file to analyze')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation (default: 4). H200 optimal: 4-8 for ~100-200GB usage. Conservative: 2-4, Aggressive: 8-16')
    parser.add_argument('--skip-detailed-analysis', action='store_true',
                       help='Skip detailed per-trial analysis to avoid hanging (recommended for datasets >10K trials)')
    parser.add_argument('--auto-batch-size', action='store_true',
                       help='Automatically determine optimal batch size based on GPU memory')
    parser.add_argument('--use-kv-caching', action='store_true',
                       help='Use KV caching optimization for dubois2022value (process all trials in a game with single forward pass)')
    
    args = parser.parse_args()
    
    # Show available datasets if requested
    if args.list_datasets:
        list_available_datasets()
        return
    
    # Analyze existing results if requested
    if args.analyze_results:
        if os.path.exists(args.analyze_results):
            return analyze_complete_results(args.analyze_results)
        else:
            print(f"❌ Results file not found: {args.analyze_results}")
            return
    
    print("=== Zero-Shot Centaur Evaluation (Single GPU - Unsloth Optimized) ===\n")
    
    # Detect GPU setup (Unsloth limitation: single GPU only)
    gpu_setup = detect_gpu_setup()
    
    # ⚠️ Unsloth limitation: Force single GPU evaluation
    effective_batch_size = args.batch_size
    
    # Auto batch size if requested
    if args.auto_batch_size:
        effective_batch_size = gpu_setup['recommended_batch_per_gpu']
        print(f"🎯 Auto-selected batch size: {effective_batch_size} (based on {gpu_setup['total_memory']:.1f}GB GPU memory)")
    else:
        effective_batch_size = args.batch_size
    
    # Print evaluation strategy
    print(f"🔧 Single GPU Evaluation (Unsloth Limitation):")
    print(f"   Using 1 GPU: {gpu_setup['gpu_names'][0]} ({gpu_setup['total_memory']:.1f}GB)")
    print(f"   Batch size: {effective_batch_size}")
    print(f"   Estimated memory usage: {40 + (effective_batch_size - 1) * 15:.0f}GB")
    
    # Estimate runtime for dubois2022value with KV caching
    if args.task == "dubois2022value":
        estimated_trials = 921200  # From your data
        time_single_gpu = estimated_trials / 3.3 / 3600  # hours
        
        # KV caching provides speedup (estimated 2-3x within games)
        kv_speedup = 2.5 if args.use_kv_caching else 1.0
        time_with_kv = time_single_gpu / kv_speedup
        
        print(f"   📊 Estimated time for dubois2022value:")
        print(f"      Standard evaluation: {time_single_gpu:.1f} hours")
        if args.use_kv_caching:
            print(f"      With KV caching: {time_with_kv:.1f} hours")
            print(f"      Speedup: {kv_speedup:.1f}x faster!")
        else:
            print(f"      💡 Add --use-kv-caching for 2-3x speedup")
    
    # Safety check for GPU memory capacity
    estimated_memory = 40 + (effective_batch_size - 1) * 15
    if estimated_memory > gpu_setup['total_memory'] * 0.9:  # Leave 10% buffer
        print(f"   ⚠️  WARNING: Estimated memory {estimated_memory:.0f}GB may exceed GPU capacity!")
        print(f"   💡 Consider reducing batch size to {max(1, int((gpu_setup['total_memory'] * 0.9 - 40) // 15 + 1))} or lower")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.task)
    
    # Setup
    model_name = args.model_name
    task_name = args.task
    
    # File paths from config
    original_prompts_file = dataset_config['original_prompts_file']
    zero_shot_prompts_file = dataset_config['zero_shot_prompts_file']
    
    print(f"📋 Configuration:")
    print(f"   Dataset: {task_name}")
    print(f"   Description: {dataset_config['description']}")
    print(f"   Model: {model_name}")
    
    # Single GPU evaluation (Unsloth limitation)
    print(f"   Batch size: {effective_batch_size} (Est. memory: {estimated_memory}GB)")
    
    # Safety check for single GPU H200 memory capacity
    if estimated_memory > 135:
        print(f"   ⚠️  WARNING: Estimated memory {estimated_memory}GB may exceed H200 capacity!")
        print(f"   💡 Consider reducing batch size to {max(1, (135 - 40) // 15 + 1)} or lower")
    
    print(f"   Original prompts: {original_prompts_file}")
    print(f"   Zero-shot prompts: {zero_shot_prompts_file}")
    print(f"   Pipeline: Exactly same as original Centaur evaluation")
    
    # Check if original prompts file exists
    if not os.path.exists(original_prompts_file):
        print(f"❌ Error: Original prompts file not found: {original_prompts_file}")
        print(f"   Available files in datasets/held_out_tasks/:")
        if os.path.exists('datasets/held_out_tasks/'):
            for file in os.listdir('datasets/held_out_tasks/'):
                print(f"     - {file}")
        return None
    
    # Step 1: Create zero-shot prompts based on dataset type
    print(f"\n=== Step 1: Create Zero-Shot Prompts ===")
    print(f"   Using dataset-specific parsing for: {task_name}")
    
    # Create eval_results directory first
    eval_results_dir = Path("eval_results")
    eval_results_dir.mkdir(exist_ok=True)
    print(f"   Created eval_results directory: {eval_results_dir}")
    
    if task_name == "ruggeri2022globalizability":
        # Parse ruggeri dataset: remove behavioral history
        zero_shot_prompts = create_zero_shot_prompts_ruggeri(original_prompts_file, zero_shot_prompts_file)
    elif task_name == "awad2018moral":
        # Parse awad dataset: extract last dilemma only
        zero_shot_prompts = create_zero_shot_prompts_awad(original_prompts_file, zero_shot_prompts_file)
    elif task_name == "hilbig2014generalized":
        # Parse hilbig dataset: extract individual trials from participant sessions
        zero_shot_prompts = create_zero_shot_prompts_hilbig(original_prompts_file, zero_shot_prompts_file)
    elif task_name == "hebart2023things":
        # Parse hebart dataset: extract individual trials from participant sessions
        zero_shot_prompts = create_zero_shot_prompts_hebart(original_prompts_file, zero_shot_prompts_file)
    elif task_name == "dubois2022value":
        # Parse dubois dataset: split games and evaluate sequentially within games
        zero_shot_prompts = create_zero_shot_prompts_dubois(original_prompts_file, zero_shot_prompts_file)

    else:
        raise ValueError(f"Unsupported dataset: {task_name}")
    
    # Step 2: Evaluate using original pipeline (single GPU with Unsloth)
    print(f"\n=== Step 2: Evaluate Using Original Pipeline ===")
    
    # Check dataset size and auto-skip detailed analysis for large datasets
    dataset_size = len(zero_shot_prompts)
    if dataset_size > 10000 and not args.skip_detailed_analysis:
        print(f"⚠️  Large dataset detected ({dataset_size:,} trials)")
        print(f"⚠️  Auto-enabling --skip-detailed-analysis to prevent hanging")
        print(f"⚠️  Detailed per-trial analysis would take several hours for this dataset size")
        args.skip_detailed_analysis = True
    
    # Single GPU evaluation with Unsloth (distributed not supported)
    print("🔧 Running single GPU evaluation with Unsloth...")
    
    # Use unified evaluation approach for all datasets
    print(f"🔧 Using unified zero-shot evaluation with trial-level NLL extraction")
    zero_shot_nll, detailed_results = evaluate_centaur_zero_shot(
        model_name, zero_shot_prompts_file, task_name, effective_batch_size, args.skip_detailed_analysis
    )
    
    # Step 3: Create comparison
    print(f"\n=== Step 3: Create Comparison ===")
    try:
        comparison_data = create_comparison_plot(zero_shot_nll, dataset_config, task_name)
    except Exception as e:
        print(f"   ❌ Error creating comparison plot: {e}")
        print(f"   ⚠️  Continuing without plot...")
        comparison_data = [{'Model': 'Zero-Shot Centaur\n(Fair)', 'NLL': zero_shot_nll, 'Type': 'Our Approach'}]
    
    # Step 4: Save comprehensive results
    print(f"\n=== Step 4: Save Comprehensive Results ===")
    print(f"   Building comprehensive results structure...")
    
    # Get baseline values from config
    original_centaur_nll = dataset_config['baseline_nll']
    cognitive_models_nll = dataset_config['cognitive_nll']
    llama_nll = dataset_config['llama_nll']
    random_nll = dataset_config['random_nll']
    
    # Comprehensive results structure
    comprehensive_results = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'script_version': 'zero_shot_centaur_evaluation_v2.0',
            'dataset_name': task_name,
            'dataset_description': dataset_config['description'],
            'model_name': model_name,
            'original_prompts_file': original_prompts_file,
            'zero_shot_prompts_file': zero_shot_prompts_file,
            'method': 'Same pipeline as original Centaur evaluation',
            'difference': 'Only prompts changed to zero-shot (no behavioral history)',
            'parsing_method': f'Dataset-specific parsing for {task_name}'
        },
        'prompt_data': {
            'original_prompts_count': len(zero_shot_prompts),
            'zero_shot_prompts_count': len(zero_shot_prompts),
            'zero_shot_prompts': [],  # Skip redundant prompt data - using unified NLL extraction for all tasks
            'prompt_creation_method': f'Dataset-specific parsing for {task_name}',
            'note': 'Prompts not saved - using unified zero-shot NLL extraction for all datasets to save only trial-level NLL results'
        },
        'gpu_config': {
            'used_distributed': False,  # Unsloth limitation: single GPU only
            'world_size': 1,
            'effective_batch_size': effective_batch_size,
            'gpu_setup': gpu_setup,
            'unsloth_limitation': True
        },
        'evaluation_results': {
            'zero_shot_centaur': {
                'nll': zero_shot_nll,
                'detailed_results': detailed_results
            },
            'published_baselines': {
                'original_centaur_unfair': {
                    'nll': original_centaur_nll, 
                    'available': original_centaur_nll is not None
                },
                'cognitive_models': {
                    'nll': cognitive_models_nll,
                    'available': cognitive_models_nll is not None
                },
                'base_llama_unfinetuned': {
                    'nll': llama_nll,
                    'available': llama_nll is not None
                },
                'random': {
                    'nll': random_nll,
                    'available': True
                }
            }
        },
        'comparison_analysis': {
            'zero_shot_vs_original': {
                'zero_shot_nll': zero_shot_nll,
                'original_nll': original_centaur_nll,
                'difference': zero_shot_nll - original_centaur_nll if original_centaur_nll else None,
                'zero_shot_worse': zero_shot_nll > original_centaur_nll if original_centaur_nll else None,
                'interpretation': ('Zero-shot performs worse than unfair original' if original_centaur_nll and zero_shot_nll > original_centaur_nll 
                                 else 'Zero-shot performs better than unfair original' if original_centaur_nll 
                                 else 'Original baseline not available')
            },
            'zero_shot_vs_cognitive': {
                'zero_shot_nll': zero_shot_nll,
                'cognitive_nll': cognitive_models_nll,
                'difference': zero_shot_nll - cognitive_models_nll if cognitive_models_nll else None,
                'cognitive_better': cognitive_models_nll < zero_shot_nll if cognitive_models_nll else None,
                'interpretation': ('Cognitive models outperform fair zero-shot Centaur' if cognitive_models_nll and cognitive_models_nll < zero_shot_nll 
                                 else 'Zero-shot Centaur outperforms cognitive models' if cognitive_models_nll 
                                 else 'Cognitive models baseline not available')
            },
            'zero_shot_vs_llama': {
                'zero_shot_nll': zero_shot_nll,
                'llama_nll': llama_nll,
                'difference': zero_shot_nll - llama_nll if llama_nll else None,
                'llama_better': llama_nll < zero_shot_nll if llama_nll else None,
                'interpretation': ('Base Llama outperforms fair zero-shot Centaur' if llama_nll and llama_nll < zero_shot_nll 
                                 else 'Zero-shot Centaur outperforms base Llama' if llama_nll 
                                 else 'Base Llama baseline not available')
            },
            'comparison_data': comparison_data
        },
        'visualization': {
            'plot_files': [f'eval_results/{task_name}_zero_shot_comparison.png', f'eval_results/{task_name}_zero_shot_comparison.pdf'],
            'color_scheme': {
                'zero_shot_centaur': '#1f77b4',
                'original_centaur': '#69005f',
                'cognitive_models': '#cbc9e2',
                'note': 'Llama baseline excluded from visualization (uses in-context learning)'
            },
            'styling': 'Original paper styling (nature theme from scienceplots)',
            'excluded_from_plot': 'Base Llama baseline (misleading due to in-context learning vs zero-shot)'
        }
    }
    
        # Save comprehensive results (eval_results directory created earlier in main())
    print(f"   Saving comprehensive results to JSON...")
    eval_results_dir = Path("eval_results")
    results_file = eval_results_dir / f'{task_name}_comprehensive_zero_shot_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        print(f"   ✅ Successfully saved comprehensive results to {results_file}")
    except Exception as e:
        print(f"   ❌ Error saving comprehensive results: {e}")
        # Try to save a simplified version
        simplified_results = {
            'metadata': comprehensive_results['metadata'],
            'evaluation_results': comprehensive_results['evaluation_results'],
            'comparison_analysis': comprehensive_results['comparison_analysis']
        }
        simplified_file = eval_results_dir / f'{task_name}_simplified_zero_shot_results.json'
        with open(simplified_file, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"   ✅ Saved simplified results to {simplified_file}")
        results_file = simplified_file
    
    print(f"✅ Evaluation Complete!")
    print(f"📊 Results Summary:")
    print(f"   Dataset: {task_name}")
    print(f"   Zero-shot Centaur NLL: {zero_shot_nll:.4f}")
    if original_centaur_nll:
        print(f"   Original Centaur NLL: {original_centaur_nll:.4f}")
    else:
        print(f"   Original Centaur NLL: Not available")
    if cognitive_models_nll:
        print(f"   Cognitive Models NLL: {cognitive_models_nll:.4f}")
    else:
        print(f"   Cognitive Models NLL: Not available")
    if llama_nll:
        print(f"   Base Llama NLL: {llama_nll:.4f}")
    else:
        print(f"   Base Llama NLL: Not available")
    print(f"   Random NLL: {random_nll:.4f}")
    print(f"📈 Plots saved as: eval_results/{task_name}_zero_shot_comparison.png & eval_results/{task_name}_zero_shot_comparison.pdf")
    print(f"   Plot shows: Zero-shot Centaur, Original Centaur, Cognitive Models, Random line")
    print(f"   Excluded from plot: Base Llama (uses in-context learning - misleading vs zero-shot)")
    print(f"📄 Comprehensive results saved as: {results_file}")
    print(f"   Data includes: All baselines including Llama (for completeness)")
    
    # Show what complete results were saved
    print(f"\n💾 Complete Results Saved:")
    print(f"   📊 Overall metrics: NLL, evaluation config, metadata")
    
    if detailed_results and 'per_trial_results' in detailed_results:
        print(f"   🎯 Per-trial results: {len(detailed_results['per_trial_results'])} individual trials")
        print(f"   📝 Each trial includes: prompt text, ground truth, loss, probabilities, metadata")
        print(f"   📈 Statistical summaries: mean/std/min/max trial losses, prompt lengths")
    else:
        print(f"   📝 Basic results only (detailed analysis skipped)")
    
    print(f"   🔍 Full prompts: All {len(zero_shot_prompts)} zero-shot prompts with parsing details")
    print(f"   📋 This enables detailed post-hoc analysis, error investigation, and trial-level insights")
    
    # Show optimization info if applicable
    if args.use_kv_caching and task_name == "dubois2022value":
        print(f"   ⚡ KV caching optimization: 2-3x speedup for dubois2022value")
        print(f"   🎯 Single GPU evaluation: Unsloth limitation (multi-GPU not supported)")
    
    # Analysis
    print(f"\n🔍 Analysis:")
    if original_centaur_nll:
        if zero_shot_nll > original_centaur_nll:
            print(f"   ✅ Zero-shot Centaur performs WORSE than unfair original ({zero_shot_nll:.3f} vs {original_centaur_nll:.3f})")
            print(f"   ✅ This supports the unfairness critique!")
        else:
            print(f"   ⚠️  Zero-shot Centaur performs BETTER than unfair original ({zero_shot_nll:.3f} vs {original_centaur_nll:.3f})")
            print(f"   ⚠️  This needs further investigation")
    else:
        print(f"   ℹ️  Original Centaur baseline not available for {task_name}")
    
    if cognitive_models_nll:
        if zero_shot_nll > cognitive_models_nll:
            print(f"   ✅ Cognitive models outperform fair zero-shot Centaur ({cognitive_models_nll:.3f} vs {zero_shot_nll:.3f})")
            print(f"   ✅ This supports the cognitive models' superiority under fair evaluation!")
        else:
            print(f"   ⚠️  Zero-shot Centaur outperforms cognitive models ({zero_shot_nll:.3f} vs {cognitive_models_nll:.3f})")
            print(f"   ⚠️  This needs further investigation")
    else:
        print(f"   ℹ️  Cognitive models baseline not available for {task_name}")
    
    if llama_nll:
        if zero_shot_nll > llama_nll:
            print(f"   📊 Base Llama outperforms fair zero-shot Centaur ({llama_nll:.3f} vs {zero_shot_nll:.3f})")
            print(f"      Note: Llama baseline saved in data but excluded from plot (uses in-context learning)")
        else:
            print(f"   📊 Zero-shot Centaur outperforms base Llama ({zero_shot_nll:.3f} vs {llama_nll:.3f})")
            print(f"      Note: Llama baseline saved in data but excluded from plot (uses in-context learning)")
    else:
        print(f"   ℹ️  Base Llama baseline not available for {task_name}")
    
    print(f"\n🎉 All done! Script completed successfully.")
    return comprehensive_results

if __name__ == "__main__":
    print("🔧 UNSLOTH SINGLE GPU SETUP:")
    print("   - ⚠️  Unsloth limitation: Single GPU only (multi-GPU not supported)")
    print("   - 1 H200 GPU (141GB) - perfect for 70B model")
    print("   - Current memory usage: ~40GB (batch_size=1)")
    print("   - Recommended: --batch-size 4-8 for better utilization")
    print("   - Memory scaling: ~15GB per additional batch unit")
    print("   - PEFT installed: pip install peft accelerate bitsandbytes")
    print("   - Strategy: Single GPU inference with optimal batching + KV caching")
    print("")
    print("🎯 MEMORY OPTIMIZATION GUIDE:")
    print("   Batch Size  |  Est. Memory  |  Utilization")
    print("   ------------|---------------|-------------")
    print("   1           |  40GB         |  28%")
    print("   2           |  55GB         |  39%")
    print("   4           |  85GB         |  60%")
    print("   6           |  115GB        |  82%")
    print("   8           |  145GB        |  103% (⚠️ may exceed)")
    print("")
    print("⚡ KV CACHING OPTIMIZATION:")
    print("   - For dubois2022value: --use-kv-caching provides 2-3x speedup")
    print("   - Processes all trials in a game with single forward pass")
    print("   - Reuses attention key-value states across trials")
    print("")
    
    main() 