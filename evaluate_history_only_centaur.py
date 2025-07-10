"""
History-Only Centaur Evaluation Script
Tests whether Centaur's performance relies on task-specific context or 
can capture deeper cognitive patterns from behavioral history alone.

This addresses the question: Does Centaur learn superficial behavioral patterns
or true cognitive generative processes?

Follows the structure of evaluate_zero_shot_centaur.py with proper batch processing
and dataset-specific parsing rules.
"""

import json
import pandas as pd
import torch
import numpy as np
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from datasets import Dataset
import argparse
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing scienceplots for original paper styling
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("⚠️ scienceplots not available - install with: pip install scienceplots")

def create_history_only_prompts_ruggeri(original_prompts_file, output_file):
    """
    Create history-only prompts for ruggeri2022globalizability: ONE prompt per participant
    
    Strategy for each participant:
    1. Extract basic instruction (remove task-specific context)
    2. Keep FULL behavioral history: "You press <<L>>. You press <<I>>. ..."
    3. Create ONE prompt per participant containing their complete sequence
    4. Evaluation will do ONE forward pass per participant and extract NLL progressively
    """
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participants")
    
    # Create history-only versions (ONE prompt per participant)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract participant-specific basic instruction (first two sentences)
        lines = prompt_text.split('\n')
        basic_instruction = ""
        sentence_count = 0
        
        for line in lines:
            if line.strip():
                line_text = line.strip()
                # Replace task-specific language with general psychological experiment language
                line_text = line_text.replace("temporal decision making task", "psychological experiment")
                line_text = line_text.replace("decision making task", "psychological experiment")
                basic_instruction += line_text + "\n"
                # Count sentences by counting periods, exclamation marks, or question marks
                sentence_count += line.count('.') + line.count('!') + line.count('?')
                if sentence_count >= 2:
                    break
        
        basic_instruction = basic_instruction.strip()
        
        # Extract choice options from the participant's specific instruction
        choice_options = []
        if "between two options" in basic_instruction:
            options_match = re.search(r'between two options ([A-Z]) and ([A-Z])', basic_instruction)
            if options_match:
                choice_options = [options_match.group(1), options_match.group(2)]
        
        if not choice_options:
            # Fallback: try to extract from the full prompt
            options_match = re.search(r'between two options ([A-Z]) and ([A-Z])', prompt_text)
            if options_match:
                choice_options = [options_match.group(1), options_match.group(2)]
        
        if not choice_options:
            print(f"   Warning: Could not extract choice options for participant {participant_data['participant']}")
            continue
        
        # Extract ALL behavioral choices from the prompt using the participant's specific options
        choice_pattern = r'You press <<([' + ''.join(choice_options) + r'])>>'
        choices = re.findall(choice_pattern, prompt_text)
        
        # Replace the reward context with the participant's specific options
        basic_instruction = basic_instruction.replace(
            "between receiving a smaller reward sooner or a larger reward later", 
            f"between <<{choice_options[0]}>> and <<{choice_options[1]}>>"
        )
        
        if len(choices) > 0:
            # Create ONE prompt containing the participant's FULL behavioral sequence
            context = basic_instruction + "\n\n"
            
            # Add ALL choices as complete behavioral history
            choice_sequence = []
            for choice in choices:
                choice_sequence.append(f"You press <<{choice}>>")
            context += "\n".join(choice_sequence)
            
            # Create history-only prompt data (ONE per participant)
            history_only_data = {
                'text': context,
                'participant': participant_data['participant'],
                'experiment': participant_data['experiment'],
                'total_choices': len(choices),
                'choice_options': choice_options,
                'basic_instruction': basic_instruction,
                'original_length': len(prompt_text),
                'history_only_length': len(context),
                'evaluation_method': 'One forward pass per participant with progressive NLL extraction'
            }
            history_only_prompts.append(history_only_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts (one per participant)")
    print(f"   Same number as original: {len(history_only_prompts) == len(original_prompts)}")
    
    # Statistics
    if history_only_prompts:
        total_choices = sum(p['total_choices'] for p in history_only_prompts)
        avg_choices_per_participant = total_choices / len(history_only_prompts)
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {avg_choices_per_participant:.1f}")
        
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total choices: {example['total_choices']}")
        print(f"   Choice options: {example['choice_options']}")
        print(f"   Text length: {len(example['text'])} characters")
        print(f"   Text preview: {example['text'][:300]}...")
    
    return history_only_prompts

def create_history_only_prompts_dubois(original_prompts_file, output_file):
    """Create history-only prompts for dubois2022value: ONE prompt per participant"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant with 400 games)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participants")
    
    # Create history-only versions (ONE prompt per participant)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract basic instruction (before first game)
        if "Game 1." in prompt_text:
            parts = prompt_text.split("Game 1.", 1)
            basic_instruction = parts[0].strip()
            games_text = "Game 1." + parts[1]
            
            # Extract ALL choices from ALL games in sequence
            import re
            game_splits = re.split(r'\n\nGame (\d+)\.', games_text)
            
            all_choices_with_context = []
            if len(game_splits) >= 1:
                # Handle Game 1
                game_1_content = game_splits[0].replace("Game 1.", "").strip()
                choices = re.findall(r'You press <<([^>]+)>>', game_1_content)
                for choice in choices:
                    all_choices_with_context.append({
                        'choice': choice,
                        'game': '1',
                        'text': f"Game 1.\nYou press <<{choice}>>"
                    })
                
                # Handle remaining games
                for i in range(1, len(game_splits) - 1, 2):
                    if i + 1 < len(game_splits):
                        game_num = game_splits[i]
                        game_content = game_splits[i + 1].strip()
                        choices = re.findall(r'You press <<([^>]+)>>', game_content)
                        for choice in choices:
                            all_choices_with_context.append({
                                'choice': choice,
                                'game': game_num,
                                'text': f"Game {game_num}.\nYou press <<{choice}>>"
                            })
            
            if len(all_choices_with_context) > 0:
                # Create ONE prompt containing the participant's FULL behavioral sequence
                context = basic_instruction + "\n\n"
                
                # Add ALL choices across ALL games as complete behavioral history
                choice_texts = []
                for choice_data in all_choices_with_context:
                    choice_texts.append(choice_data['text'])
                context += "\n\n".join(choice_texts)
                
                # Create history-only prompt data (ONE per participant)
                history_only_data = {
                    'text': context,
                    'participant': participant_data['participant'],
                    'experiment': participant_data['experiment'],
                    'total_choices': len(all_choices_with_context),
                    'num_games': len(set(c['game'] for c in all_choices_with_context)),
                    'basic_instruction': basic_instruction,
                    'original_length': len(prompt_text),
                    'history_only_length': len(context),
                    'evaluation_method': 'One forward pass per participant with progressive NLL extraction'
                }
                history_only_prompts.append(history_only_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts (one per participant)")
    print(f"   Same number as original: {len(history_only_prompts) == len(original_prompts)}")
    
    # Statistics
    if history_only_prompts:
        total_choices = sum(p['total_choices'] for p in history_only_prompts)
        avg_choices_per_participant = total_choices / len(history_only_prompts)
        total_games = sum(p['num_games'] for p in history_only_prompts)
        avg_games_per_participant = total_games / len(history_only_prompts)
        
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {avg_choices_per_participant:.1f}")
        print(f"   Average games per participant: {avg_games_per_participant:.1f}")
        
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total choices: {example['total_choices']}")
        print(f"   Games: {example['num_games']}")
        print(f"   Text length: {len(example['text'])} characters")
        print(f"   Text preview: {example['text'][:300]}...")
    
    return history_only_prompts

def create_history_only_prompts_wu2018(original_prompts_file, output_file):
    """Create history-only prompts for wu2018generalisation_exp1: ONE prompt per participant"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant with multiple environments)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participants")
    
    # Create history-only versions (ONE prompt per participant)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract basic instruction (first two sentences, modify "to explore" -> "to choose")
        lines = prompt_text.split('\n')
        basic_instruction = ""
        sentence_count = 0
        
        for line in lines:
            if line.strip():
                line_text = line.strip()
                # Replace task-specific language with general psychological experiment language
                line_text = line_text.replace("to explore", "to choose")
                basic_instruction += line_text + "\n"
                # Count sentences by counting periods, exclamation marks, or question marks
                sentence_count += line.count('.') + line.count('!') + line.count('?')
                if sentence_count >= 2:
                    break
        
        basic_instruction = basic_instruction.strip()
        
        # Extract ALL choices from ALL environments in sequence
        import re
        environment_splits = re.split(r'\n\nEnvironment number (\d+):', prompt_text)
        
        all_choices_with_context = []
        if len(environment_splits) >= 2:
            # Process each environment and collect all choices in order
            for i in range(1, len(environment_splits) - 1, 2):
                env_num = environment_splits[i]
                env_content = environment_splits[i + 1].strip()
                
                # Extract choices from this environment
                choices = re.findall(r'You press <<(\d+)>>', env_content)
                
                # Add each choice with its environment context
                for choice in choices:
                    all_choices_with_context.append({
                        'choice': choice,
                        'environment': env_num,
                        'text': f"Environment number {env_num}:\nYou press <<{choice}>>"
                    })
        
        if len(all_choices_with_context) > 0:
            # Create ONE prompt containing the participant's FULL behavioral sequence
            context = basic_instruction + "\n\n"
            
            # Add ALL choices across ALL environments as complete behavioral history
            choice_texts = []
            for choice_data in all_choices_with_context:
                choice_texts.append(choice_data['text'])
            context += "\n\n".join(choice_texts)
            
            # Create history-only prompt data (ONE per participant)
            history_only_data = {
                'text': context,
                'participant': participant_data['participant'],
                'experiment': participant_data['experiment'],
                'total_choices': len(all_choices_with_context),
                'num_environments': len(set(c['environment'] for c in all_choices_with_context)),
                'basic_instruction': basic_instruction,
                'original_length': len(prompt_text),
                'history_only_length': len(context),
                'evaluation_method': 'One forward pass per participant with progressive NLL extraction'
            }
            history_only_prompts.append(history_only_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts (one per participant)")
    print(f"   Same number as original: {len(history_only_prompts) == len(original_prompts)}")
    
    # Statistics
    if history_only_prompts:
        total_choices = sum(p['total_choices'] for p in history_only_prompts)
        avg_choices_per_participant = total_choices / len(history_only_prompts)
        total_environments = sum(p['num_environments'] for p in history_only_prompts)
        avg_environments_per_participant = total_environments / len(history_only_prompts)
        
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {avg_choices_per_participant:.1f}")
        print(f"   Average environments per participant: {avg_environments_per_participant:.1f}")
        
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total choices: {example['total_choices']}")
        print(f"   Environments: {example['num_environments']}")
        print(f"   Text length: {len(example['text'])} characters")
        print(f"   Text preview: {example['text'][:300]}...")
    
    return history_only_prompts

def create_history_only_prompts_collsioo(original_prompts_file, output_file):
    """Create history-only prompts for collsiöö2023MCPL_exp1 - single prompt per participant with all choices for progressive NLL extraction"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant session)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participant sessions")
    
    # Create history-only versions (one prompt per participant with all choices)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract basic domain context + choice space (ultra-minimal: ONLY Caldionine task and choice space)
        lines = prompt_text.split('\n')
        basic_context_lines = []
        
        # Find the basic domain context (what we're estimating)
        for line in lines:
            line = line.strip()
            if line.startswith("Your task is to estimate") and "Caldionine" in line:
                # Keep basic domain context but remove causal explanations
                if "based on" in line:
                    # Remove the "based on X and Y" part
                    basic_part = line.split("based on")[0].strip()
                    basic_context_lines.append(basic_part)
                else:
                    basic_context_lines.append(line)
                break
        
        # Find ONLY the Caldionine choice space (not Progladine/Amalydine)
        for line in lines:
            line = line.strip()
            if "Caldionine can take" in line:
                basic_context_lines.append(line)
                break
        
        # Fallback if not found
        if not basic_context_lines:
            basic_context_lines = [
                "Your task is to estimate the blood concentration of the hormone Caldionine.",
                "Caldionine can take nine values (1, 2, 3, 4, 5, 6, 7, 8, 9)."
            ]
        
        minimal_context = '\n'.join(basic_context_lines)
        
        # Extract all trials from the original prompt
        import re
        trial_pattern = r'Progladine: ([^.]*?)\. Amalydine: ([^.]*?)\. You say that the Caldionine concentration is <<([^>]+)>>\. That is ([^.]*?)\. The correct concentration of Caldionine is ([^.]*?)\.'
        trials = re.findall(trial_pattern, prompt_text)
        
        if not trials:
            # Try alternative pattern for different format
            alt_pattern = r'Progladine: ([^.]*?)\. Amalydine: ([^.]*?)\. You say that the Caldionine concentration is <<([^>]+)>>'
            trials = re.findall(alt_pattern, prompt_text)
            # For alternative pattern, we don't have feedback info
            trials = [(prog, amal, choice, "unknown", "unknown") for prog, amal, choice in trials]
        
        print(f"   Participant {participant_data['participant']}: found {len(trials)} trials")
        
        if trials:
            # Build ultra-minimal history prompt: minimal context + choice sequence only (NO state info, NO feedback)
            full_history_prompt = minimal_context + "\n\n"
            
            choice_positions = []  # Track where each choice token appears for NLL extraction
            
            for i, (prog, amal, choice, feedback, correct) in enumerate(trials):
                # Ultra-minimal format: ONLY the choice statements, no context or feedback
                choice_statement = f"You say that the Caldionine concentration is <<{choice}>>."
                
                # Mark the position of this choice for later NLL extraction
                choice_start_pos = len(full_history_prompt) + len("You say that the Caldionine concentration is <<")
                choice_positions.append({
                    'choice_index': i,
                    'choice': choice,
                    'choice_start_pos': choice_start_pos,
                    'progladine': prog,  # Keep for analysis but not in prompt
                    'amalydine': amal,   # Keep for analysis but not in prompt
                    'feedback': feedback,
                    'correct': correct
                })
                
                full_history_prompt += choice_statement + "\n"
            
            # Create history-only prompt data (one per participant with all choices)
            history_only_data = {
                'text': full_history_prompt.strip(),
                'participant': participant_data['participant'],
                'experiment': participant_data['experiment'],
                'num_trials': len(trials),
                'choice_positions': choice_positions,  # For progressive NLL extraction
                'basic_instruction': minimal_context,
                'trials': [
                    {
                        'trial_number': i + 1,
                        'progladine': prog,
                        'amalydine': amal,
                        'choice': choice,
                        'feedback': feedback,
                        'correct': correct
                    }
                    for i, (prog, amal, choice, feedback, correct) in enumerate(trials)
                ],
                'original_length': len(prompt_text),
                'history_only_length': len(full_history_prompt)
            }
            history_only_prompts.append(history_only_data)
        else:
            # Fallback - keep as is if no standard format found
            history_only_prompts.append(participant_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        total_choices = sum(p.get('num_trials', 0) for p in history_only_prompts)
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {total_choices / len(original_prompts):.1f}")
    
    # Statistics
    if history_only_prompts:
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts if 'history_only_length' in p and 'original_length' in p])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total trials: {example.get('num_trials', 'N/A')}")
        print(f"   Context length: {len(example['text'])} characters")
        print(f"   Choice positions: {len(example.get('choice_positions', []))} choices marked for NLL extraction")
        print(f"   Preview: {example['text'][:300]}...")
    
    return history_only_prompts

def create_history_only_prompts_collsioo_multi(experiment_files, output_file):
    """Create history-only prompts for collsiöö2023MCPL exp1, exp2, exp3 - single prompt per participant with all choices for progressive NLL extraction"""
    print(f"📝 Creating history-only prompts from: {experiment_files}")
    
    all_prompts = []
    for exp_idx, exp_file in enumerate(experiment_files):
        exp_tag = f"exp{exp_idx+1}"
        print(f"   Processing {exp_tag}: {exp_file}")
        
        with open(exp_file, 'r') as f:
            for line in f:
                participant_data = json.loads(line)
                prompt_text = participant_data['text']
                
                # Extract basic domain context + choice space (ultra-minimal: ONLY Caldionine task and choice space)
                lines = prompt_text.split('\n')
                basic_context_lines = []
                
                # Find the basic domain context (what we're estimating)
                for line_ in lines:
                    line_ = line_.strip()
                    if line_.startswith("Your task is to estimate") and "Caldionine" in line_:
                        # Keep basic domain context but remove causal explanations
                        if "based on" in line_:
                            # Remove the "based on X and Y" part
                            basic_part = line_.split("based on")[0].strip()
                            basic_context_lines.append(basic_part)
                        else:
                            basic_context_lines.append(line_)
                        break
                
                # Find ONLY the Caldionine choice space (not Progladine/Amalydine)
                for line_ in lines:
                    line_ = line_.strip()
                    if "Caldionine can take" in line_:
                        basic_context_lines.append(line_)
                        break
                
                # Fallback if not found
                if not basic_context_lines:
                    basic_context_lines = [
                        "Your task is to estimate the blood concentration of the hormone Caldionine.",
                        "Caldionine can take nine values (1, 2, 3, 4, 5, 6, 7, 8, 9)."
                    ]
                
                minimal_context = '\n'.join(basic_context_lines)
                
                # Extract all trials
                import re
                trial_pattern = r'Progladine: ([^.]*?)\. Amalydine: ([^.]*?)\. You say that the Caldionine concentration is <<([^>]+)>>\. That is ([^.]*?)\. The correct concentration of Caldionine is ([^.]*?)\.'
                trials = re.findall(trial_pattern, prompt_text)
                
                if not trials:
                    alt_pattern = r'Progladine: ([^.]*?)\. Amalydine: ([^.]*?)\. You say that the Caldionine concentration is <<([^>]+)>>'
                    trials = re.findall(alt_pattern, prompt_text)
                    trials = [(prog, amal, choice, "unknown", "unknown") for prog, amal, choice in trials]
                
                if trials:
                    # Build ultra-minimal history prompt: minimal context + choice sequence only (NO state info, NO feedback)
                    full_history_prompt = minimal_context + "\n\n"
                    
                    choice_positions = []  # Track where each choice token appears for NLL extraction
                    
                    for i, (prog, amal, choice, feedback, correct) in enumerate(trials):
                        # Ultra-minimal format: ONLY the choice statements, no context or feedback
                        choice_statement = f"You say that the Caldionine concentration is <<{choice}>>."
                        
                        # Mark the position of this choice for later NLL extraction
                        choice_start_pos = len(full_history_prompt) + len("You say that the Caldionine concentration is <<")
                        choice_positions.append({
                            'choice_index': i,
                            'choice': choice,
                            'choice_start_pos': choice_start_pos,
                            'progladine': prog,  # Keep for analysis but not in prompt
                            'amalydine': amal,   # Keep for analysis but not in prompt
                            'feedback': feedback,
                            'correct': correct
                        })
                        
                        full_history_prompt += choice_statement + "\n"
                    
                    # Create history-only prompt data (one per participant with all choices)
                    history_only_data = {
                        'text': full_history_prompt.strip(),
                        'participant': participant_data['participant'],
                        'experiment': participant_data['experiment'],
                        'exp_tag': exp_tag,
                        'num_trials': len(trials),
                        'choice_positions': choice_positions,  # For progressive NLL extraction
                        'basic_instruction': minimal_context,
                        'trials': [
                            {
                                'trial_number': i + 1,
                                'progladine': prog,
                                'amalydine': amal,
                                'choice': choice,
                                'feedback': feedback,
                                'correct': correct
                            }
                            for i, (prog, amal, choice, feedback, correct) in enumerate(trials)
                        ],
                        'original_length': len(prompt_text),
                        'history_only_length': len(full_history_prompt)
                    }
                    all_prompts.append(history_only_data)
                else:
                    # Fallback - keep as is if no standard format found
                    participant_data['exp_tag'] = exp_tag
                    all_prompts.append(participant_data)
    
    print(f"   Created {len(all_prompts)} participant prompts from {len(experiment_files)} experiments")
    if len(all_prompts) > 0:
        total_choices = sum(p.get('num_trials', 0) for p in all_prompts)
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {total_choices / len(all_prompts):.1f}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in all_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if all_prompts:
        example = all_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Experiment: {example.get('exp_tag', 'N/A')}")
        print(f"   Total trials: {example.get('num_trials', 'N/A')}")
        print(f"   Context length: {len(example['text'])} characters")
        print(f"   Choice positions: {len(example.get('choice_positions', []))} choices marked for NLL extraction")
        print(f"   Preview: {example['text'][:300]}...")
    
    return all_prompts

def create_history_only_prompts_hilbig(original_prompts_file, output_file):
    """Create history-only prompts for hilbig2014generalized - ultra-minimal context with choice sequence only"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant session)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participant sessions")
    
    # Create history-only versions (one prompt per participant with all choices)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract ultra-minimal context: ONLY the first sentence about choice options
        lines = prompt_text.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Extract ONLY the first sentence (before first period)
        sentences = first_line.split('.')
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip() + "."
            
            # Extract the choice option labels dynamically from the first sentence
            import re
            # Pattern to find "labeled X and Y" or similar
            label_match = re.search(r'labeled ([A-Z]+) and ([A-Z]+)', first_sentence)
            if label_match:
                option1, option2 = label_match.groups()
                minimal_context = f"You are repeatedly presented with two options, labeled {option1} and {option2}."
            else:
                # Fallback: use the first sentence as-is if it's about two options
                if "two options" in first_sentence:
                    minimal_context = first_sentence
                else:
                    minimal_context = "You are repeatedly presented with two options, labeled P and G."
        else:
            # Ultimate fallback
            minimal_context = "You are repeatedly presented with two options, labeled P and G."
        
        # Extract ALL choices from the participant session using dynamic pattern
        import re
        # Extract choice options from the minimal context
        label_match = re.search(r'labeled ([A-Z]+) and ([A-Z]+)', minimal_context)
        if label_match:
            option1, option2 = label_match.groups()
            choice_pattern = f'You press <<([{option1}{option2}])>>'
        else:
            # Fallback pattern
            choice_pattern = r'You press <<([A-Z])>>'
        
        choices = re.findall(choice_pattern, prompt_text)
        
        print(f"   Participant {participant_data.get('participant', 'unknown')}: found {len(choices)} choices")
        
        if choices:
            # Build ultra-minimal history prompt: minimal context + choice sequence only (NO product ratings, NO expert info)
            full_history_prompt = minimal_context + "\n\n"
            
            choice_positions = []  # Track where each choice token appears for NLL extraction
            
            for i, choice in enumerate(choices):
                # Ultra-minimal format: ONLY the choice statements, no context or ratings
                choice_statement = f"You press <<{choice}>>."
                
                # Mark the position of this choice for later NLL extraction
                choice_start_pos = len(full_history_prompt) + len("You press <<")
                choice_positions.append({
                    'choice_index': i,
                    'choice': choice,
                    'choice_start_pos': choice_start_pos
                })
                
                full_history_prompt += choice_statement + "\n"
            
            # Create history-only prompt data (one per participant with all choices)
            history_only_data = {
                'text': full_history_prompt.strip(),
                'participant': participant_data['participant'],
                'experiment': participant_data['experiment'],
                'num_trials': len(choices),
                'choice_positions': choice_positions,  # For progressive NLL extraction
                'basic_instruction': minimal_context,
                'choices': [
                    {
                        'trial_number': i + 1,
                        'choice': choice
                    }
                    for i, choice in enumerate(choices)
                ],
                'original_length': len(prompt_text),
                'history_only_length': len(full_history_prompt)
            }
            history_only_prompts.append(history_only_data)
        else:
            # Fallback - keep as is if no choices found
            history_only_prompts.append(participant_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        total_choices = sum(p.get('num_trials', 0) for p in history_only_prompts)
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {total_choices / len(original_prompts):.1f}")
    
    # Statistics
    if history_only_prompts:
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts if 'history_only_length' in p and 'original_length' in p])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total trials: {example.get('num_trials', 'N/A')}")
        print(f"   Context length: {len(example['text'])} characters")
        print(f"   Choice positions: {len(example.get('choice_positions', []))} choices marked for NLL extraction")
        print(f"   Context reduction: {(1 - example.get('history_only_length', 0) / example.get('original_length', 1)):.1%}")
        print(f"   Preview: {example['text'][:300]}...")
    
    return history_only_prompts

def create_history_only_prompts_hebart(original_prompts_file, output_file):
    """Create history-only prompts for hebart2023things - ultra-minimal context with choice sequence only"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    
    # Load original prompts (each line = one participant session)
    original_prompts = []
    with open(original_prompts_file, 'r') as f:
        for line in f:
            original_prompts.append(json.loads(line))
    
    print(f"   Loaded {len(original_prompts)} participant sessions")
    
    # Create history-only versions (one prompt per participant with all choices)
    history_only_prompts = []
    
    for participant_data in original_prompts:
        prompt_text = participant_data['text']
        
        # Extract ultra-minimal context: ONLY the first sentence about the task
        lines = prompt_text.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Extract ONLY the first sentence (before first period)
        sentences = first_line.split('.')
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip() + "."
            minimal_context = first_sentence
        else:
            # Ultimate fallback
            minimal_context = "You are presented with three objects."
        
        # Extract ALL choices from the participant session
        import re
        trial_pattern = r'([A-Z]): ([^,]+), ([A-Z]): ([^,]+), and ([A-Z]): ([^.]+)\. You press <<([A-Z])>>\.'
        trials = re.finditer(trial_pattern, prompt_text)
        
        choices = []
        trial_count = 0
        for trial_match in trials:
            choice = trial_match.group(7)  # The choice letter (A, B, or C) - FIXED: was group(6)
            choices.append(choice)
            trial_count += 1
        
        print(f"   Participant {participant_data.get('participant', 'unknown')}: found {len(choices)} choices")
        
        if choices:
            # Build ultra-minimal history prompt: minimal context + choice sequence only (NO object descriptions)
            full_history_prompt = minimal_context + "\n\n"
            
            choice_positions = []  # Track where each choice token appears for NLL extraction
            
            for i, choice in enumerate(choices):
                # Ultra-minimal format: ONLY the choice statements, no object descriptions
                choice_statement = f"You press <<{choice}>>."
                
                # Mark the position of this choice for later NLL extraction
                choice_start_pos = len(full_history_prompt) + len("You press <<")
                choice_positions.append({
                    'choice_index': i,
                    'choice': choice,
                    'choice_start_pos': choice_start_pos
                })
                
                full_history_prompt += choice_statement + "\n"
            
            # Create history-only prompt data (one per participant with all choices)
            history_only_data = {
                'text': full_history_prompt.strip(),
                'participant': participant_data['participant'],
                'experiment': participant_data['experiment'],
                'num_trials': len(choices),
                'choice_positions': choice_positions,  # For progressive NLL extraction
                'basic_instruction': minimal_context,
                'choices': [
                    {
                        'trial_number': i + 1,
                        'choice': choice
                    }
                    for i, choice in enumerate(choices)
                ],
                'original_length': len(prompt_text),
                'history_only_length': len(full_history_prompt)
            }
            history_only_prompts.append(history_only_data)
        else:
            # Fallback - keep as is if no choices found
            history_only_prompts.append(participant_data)
    
    print(f"   Created {len(history_only_prompts)} participant prompts from {len(original_prompts)} participants")
    if len(original_prompts) > 0:
        total_choices = sum(p.get('num_trials', 0) for p in history_only_prompts)
        print(f"   Total choices across all participants: {total_choices}")
        print(f"   Average choices per participant: {total_choices / len(original_prompts):.1f}")
    
    # Statistics
    if history_only_prompts:
        avg_reduction = np.mean([1 - p['history_only_length'] / p['original_length'] for p in history_only_prompts if 'history_only_length' in p and 'original_length' in p])
        print(f"   Average context reduction: {avg_reduction:.1%}")
    
    # Save history-only prompts
    with open(output_file, 'w') as f:
        for prompt in history_only_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"   Saved to: {output_file}")
    
    # Show example
    if history_only_prompts:
        example = history_only_prompts[0]
        print(f"\n📝 Example participant prompt:")
        print(f"   Participant: {example['participant']}")
        print(f"   Total trials: {example.get('num_trials', 'N/A')}")
        print(f"   Context length: {len(example['text'])} characters")
        print(f"   Choice positions: {len(example.get('choice_positions', []))} choices marked for NLL extraction")
        print(f"   Context reduction: {(1 - example.get('history_only_length', 0) / example.get('original_length', 1)):.1%}")
        print(f"   Preview: {example['text'][:300]}...")
    
    return history_only_prompts

def evaluate_collsioo_with_progressive_nll(model, tokenizer, dataset_list, condition_name, batch_size=4):
    """
    Specialized evaluation for collsioo datasets with progressive NLL extraction.
    Each participant prompt contains all choices, and we extract NLL for each choice token progressively.
    """
    print(f"🔍 Evaluating {condition_name} with progressive NLL extraction")
    print(f"   Processing {len(dataset_list)} participant prompts")
    
    device = next(model.parameters()).device
    per_trial_results = []
    all_individual_token_nlls = []
    
    for participant_idx, participant_data in enumerate(dataset_list):
        prompt_text = participant_data['text']
        choice_positions = participant_data.get('choice_positions', [])
        
        if not choice_positions:
            print(f"   Warning: No choice positions for participant {participant_idx}, skipping...")
            continue
        
        print(f"   Processing participant {participant_data.get('participant', participant_idx)}: {len(choice_positions)} choices")
        
        # Tokenize the full prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Single forward pass for the entire participant prompt
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Extract NLL for each choice token progressively
            for choice_info in choice_positions:
                choice_index = choice_info['choice_index']
                choice = choice_info['choice']
                choice_start_pos = choice_info['choice_start_pos']
                
                # Find the choice tokens in the tokenized sequence
                # We need to find where "<<{choice}>>" appears in the tokenized text
                if choice is None or choice == "":
                    print(f"   Warning: Empty choice for choice {choice_index}, skipping...")
                    choice_nll = float('inf')
                    individual_token_nlls = []
                    # Store detailed results for this choice
                    trial_result = {
                        'trial_index': len(per_trial_results),
                        'participant': participant_data.get('participant'),
                        'experiment': participant_data.get('experiment'),
                        'exp_tag': participant_data.get('exp_tag'),
                        'choice_index': choice_index,
                        'total_choices_in_participant': len(choice_positions),
                        'ground_truth_choice': choice,
                        'trial_nll': choice_nll,
                        'individual_token_nlls': individual_token_nlls,
                        'has_history': choice_index > 0,  # First choice has no history
                        'num_history_choices': choice_index,  # Number of previous choices
                        'progladine': choice_info.get('progladine'),
                        'amalydine': choice_info.get('amalydine'),
                        'feedback': choice_info.get('feedback'),
                        'correct': choice_info.get('correct'),
                        'prompt_length_chars': len(prompt_text),
                        'dataset': condition_name
                    }
                    per_trial_results.append(trial_result)
                    continue
                
                choice_pattern = f"<<{choice}>>"
                
                # Search for the choice pattern in the text around the expected position
                search_start = max(0, choice_start_pos - 50)
                search_end = min(len(prompt_text), choice_start_pos + 50)
                search_text = prompt_text[search_start:search_end]
                
                if choice_pattern in search_text:
                    # Find the actual position in the text
                    pattern_pos_in_search = search_text.find(choice_pattern)
                    pattern_pos_in_full = search_start + pattern_pos_in_search
                    
                    # Tokenize text up to the choice to find token position
                    text_before_choice = prompt_text[:pattern_pos_in_full + 2]  # Include "<<"
                    tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
                    choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1
                    
                    # Tokenize just the choice content
                    choice_tokens = tokenizer(choice, return_tensors="pt", add_special_tokens=False)
                    choice_token_ids = choice_tokens['input_ids'][0]
                    
                    # Extract logits for predicting the choice tokens
                    if choice_start_token_pos + len(choice_token_ids) <= len(logits):
                        choice_logits = logits[choice_start_token_pos:choice_start_token_pos + len(choice_token_ids)]
                        
                        # Compute NLL for each choice token
                        individual_token_nlls = []
                        for i, token_id in enumerate(choice_token_ids):
                            if i < len(choice_logits):
                                token_logits = choice_logits[i]
                                log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                                token_nll = -log_probs[token_id].item()
                                individual_token_nlls.append(token_nll)
                        
                        # Average NLL for this choice
                        if individual_token_nlls:
                            choice_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                            all_individual_token_nlls.extend(individual_token_nlls)
                        else:
                            choice_nll = float('inf')
                    else:
                        choice_nll = float('inf')
                        individual_token_nlls = []
                else:
                    print(f"   Warning: Could not find choice pattern for choice {choice_index} (choice='{choice}')")
                    choice_nll = float('inf')
                    individual_token_nlls = []
                
                # Store detailed results for this choice
                trial_result = {
                    'trial_index': len(per_trial_results),
                    'participant': participant_data.get('participant'),
                    'experiment': participant_data.get('experiment'),
                    'exp_tag': participant_data.get('exp_tag'),
                    'choice_index': choice_index,
                    'total_choices_in_participant': len(choice_positions),
                    'ground_truth_choice': choice,
                    'trial_nll': choice_nll,
                    'individual_token_nlls': individual_token_nlls,
                    'has_history': choice_index > 0,  # First choice has no history
                    'num_history_choices': choice_index,  # Number of previous choices
                    'progladine': choice_info.get('progladine'),
                    'amalydine': choice_info.get('amalydine'),
                    'feedback': choice_info.get('feedback'),
                    'correct': choice_info.get('correct'),
                    'prompt_length_chars': len(prompt_text),
                    'dataset': condition_name
                }
                per_trial_results.append(trial_result)
        
        # Progress update
        if (participant_idx + 1) % 10 == 0:
            recent_nlls = [r['trial_nll'] for r in per_trial_results[-10:] if r['trial_nll'] != float('inf')]
            avg_recent_nll = sum(recent_nlls) / len(recent_nlls) if recent_nlls else float('inf')
            print(f"     Processed {participant_idx + 1}/{len(dataset_list)} participants... Recent avg NLL: {avg_recent_nll:.3f}")
    
    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(all_individual_token_nlls) / len(all_individual_token_nlls) if all_individual_token_nlls else float('inf')
    
    print(f"   ✅ Progressive NLL extraction complete")
    print(f"   🎯 Overall NLL: {overall_nll:.4f}")
    print(f"   🔍 Valid choices: {len(valid_trial_nlls)}/{len(per_trial_results)}")
    print(f"   📊 Total tokens evaluated: {len(all_individual_token_nlls)}")
    
    # Create basic results structure
    basic_results = {
        'condition': condition_name,
        'nll': overall_nll,
        'num_samples': len(valid_trial_nlls),
        'total_choices_processed': len(per_trial_results),
        'total_participants_processed': len(dataset_list),
        'evaluation_method': 'Progressive NLL extraction from full participant prompts',
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create detailed results structure  
    detailed_results = {
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_trial_nlls),
            'total_tokens_evaluated': len(all_individual_token_nlls),
            'overall_nll': overall_nll,
            'raw_token_nlls': all_individual_token_nlls,
            'trials_with_history': len([r for r in per_trial_results if r.get('has_history', False)]),
            'trials_without_history': len([r for r in per_trial_results if not r.get('has_history', False)])
        },
        'collection_method': 'Progressive NLL extraction with single forward pass per participant',
        'condition': condition_name
    }
    
    return overall_nll, len(valid_trial_nlls), basic_results, detailed_results

def evaluate_hilbig_with_progressive_nll(model, tokenizer, dataset_list, condition_name, batch_size=4):
    """
    Specialized evaluation for hilbig2014generalized datasets with progressive NLL extraction.
    Each participant prompt contains all choices, and we extract NLL for each choice token progressively.
    """
    print(f"🔍 Evaluating {condition_name} with progressive NLL extraction")
    print(f"   Processing {len(dataset_list)} participant prompts")
    
    device = next(model.parameters()).device
    per_trial_results = []
    all_individual_token_nlls = []
    
    for participant_idx, participant_data in enumerate(dataset_list):
        prompt_text = participant_data['text']
        choice_positions = participant_data.get('choice_positions', [])
        
        if not choice_positions:
            print(f"   Warning: No choice positions for participant {participant_idx}, skipping...")
            continue
        
        print(f"   Processing participant {participant_data.get('participant', participant_idx)}: {len(choice_positions)} choices")
        
        # Tokenize the full prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Single forward pass for the entire participant prompt
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Extract NLL for each choice token progressively
            for choice_info in choice_positions:
                choice_index = choice_info['choice_index']
                choice = choice_info['choice']
                choice_start_pos = choice_info['choice_start_pos']
                
                # Find the choice tokens in the tokenized sequence
                # We need to find where "<<{choice}>>" appears in the tokenized text
                if choice is None or choice == "":
                    print(f"   Warning: Empty choice for choice {choice_index}, skipping...")
                    choice_nll = float('inf')
                    individual_token_nlls = []
                    # Store detailed results for this choice
                    trial_result = {
                        'trial_index': len(per_trial_results),
                        'participant': participant_data.get('participant'),
                        'experiment': participant_data.get('experiment'),
                        'choice_index': choice_index,
                        'total_choices_in_participant': len(choice_positions),
                        'ground_truth_choice': choice,
                        'trial_nll': choice_nll,
                        'individual_token_nlls': individual_token_nlls,
                        'has_history': choice_index > 0,  # First choice has no history
                        'num_history_choices': choice_index,  # Number of previous choices
                        'prompt_length_chars': len(prompt_text),
                        'dataset': condition_name
                    }
                    per_trial_results.append(trial_result)
                    continue
                
                choice_pattern = f"<<{choice}>>"
                
                # Search for the choice pattern in the text around the expected position
                search_start = max(0, choice_start_pos - 50)
                search_end = min(len(prompt_text), choice_start_pos + 50)
                search_text = prompt_text[search_start:search_end]
                
                if choice_pattern in search_text:
                    # Find the actual position in the text
                    pattern_pos_in_search = search_text.find(choice_pattern)
                    pattern_pos_in_full = search_start + pattern_pos_in_search
                    
                    # Tokenize text up to the choice to find token position
                    text_before_choice = prompt_text[:pattern_pos_in_full + 2]  # Include "<<"
                    tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
                    choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1
                    
                    # Tokenize just the choice content
                    choice_tokens = tokenizer(choice, return_tensors="pt", add_special_tokens=False)
                    choice_token_ids = choice_tokens['input_ids'][0]
                    
                    # Extract logits for predicting the choice tokens
                    if choice_start_token_pos + len(choice_token_ids) <= len(logits):
                        choice_logits = logits[choice_start_token_pos:choice_start_token_pos + len(choice_token_ids)]
                        
                        # Compute NLL for each choice token
                        individual_token_nlls = []
                        for i, token_id in enumerate(choice_token_ids):
                            if i < len(choice_logits):
                                token_logits = choice_logits[i]
                                log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                                token_nll = -log_probs[token_id].item()
                                individual_token_nlls.append(token_nll)
                        
                        # Average NLL for this choice
                        if individual_token_nlls:
                            choice_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                            all_individual_token_nlls.extend(individual_token_nlls)
                        else:
                            choice_nll = float('inf')
                    else:
                        choice_nll = float('inf')
                        individual_token_nlls = []
                else:
                    print(f"   Warning: Could not find choice pattern for choice {choice_index} (choice='{choice}')")
                    choice_nll = float('inf')
                    individual_token_nlls = []
                
                # Store detailed results for this choice
                trial_result = {
                    'trial_index': len(per_trial_results),
                    'participant': participant_data.get('participant'),
                    'experiment': participant_data.get('experiment'),
                    'choice_index': choice_index,
                    'total_choices_in_participant': len(choice_positions),
                    'ground_truth_choice': choice,
                    'trial_nll': choice_nll,
                    'individual_token_nlls': individual_token_nlls,
                    'has_history': choice_index > 0,  # First choice has no history
                    'num_history_choices': choice_index,  # Number of previous choices
                    'prompt_length_chars': len(prompt_text),
                    'dataset': condition_name
                }
                per_trial_results.append(trial_result)
        
        # Progress update
        if (participant_idx + 1) % 10 == 0:
            recent_nlls = [r['trial_nll'] for r in per_trial_results[-10:] if r['trial_nll'] != float('inf')]
            avg_recent_nll = sum(recent_nlls) / len(recent_nlls) if recent_nlls else float('inf')
            print(f"     Processed {participant_idx + 1}/{len(dataset_list)} participants... Recent avg NLL: {avg_recent_nll:.3f}")
    
    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(all_individual_token_nlls) / len(all_individual_token_nlls) if all_individual_token_nlls else float('inf')
    
    print(f"   ✅ Progressive NLL extraction complete")
    print(f"   🎯 Overall NLL: {overall_nll:.4f}")
    print(f"   🔍 Valid choices: {len(valid_trial_nlls)}/{len(per_trial_results)}")
    print(f"   📊 Total tokens evaluated: {len(all_individual_token_nlls)}")
    
    # Create basic results structure
    basic_results = {
        'condition': condition_name,
        'nll': overall_nll,
        'num_samples': len(valid_trial_nlls),
        'total_choices_processed': len(per_trial_results),
        'total_participants_processed': len(dataset_list),
        'evaluation_method': 'Progressive NLL extraction from full participant prompts (hilbig2014generalized)',
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create detailed results structure  
    detailed_results = {
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_trial_nlls),
            'total_tokens_evaluated': len(all_individual_token_nlls),
            'overall_nll': overall_nll,
            'raw_token_nlls': all_individual_token_nlls,
            'trials_with_history': len([r for r in per_trial_results if r.get('has_history', False)]),
            'trials_without_history': len([r for r in per_trial_results if not r.get('has_history', False)])
        },
        'collection_method': 'Progressive NLL extraction with single forward pass per participant (hilbig2014generalized)',
        'condition': condition_name
    }
    
    return overall_nll, len(valid_trial_nlls), basic_results, detailed_results

def evaluate_wu2018_with_progressive_nll(model, tokenizer, dataset_list, condition_name, batch_size=4):
    """
    Simple progressive NLL extraction for wu2018generalisation_exp1.
    One forward pass per participant, extract NLL for each choice token progressively.
    Returns simple trial-level likelihood list like ruggeri2022globalizability format.
    """
    print(f"🔍 Evaluating {condition_name} with progressive NLL extraction")
    print(f"   Processing {len(dataset_list)} participant prompts")
    
    device = next(model.parameters()).device
    per_trial_results = []
    all_individual_token_nlls = []
    
    for participant_idx, participant_data in enumerate(dataset_list):
        prompt_text = participant_data['text']
        
        # Extract choice patterns from the prompt
        import re
        choice_pattern = r'You press <<(\d+)>>'
        choice_matches = list(re.finditer(choice_pattern, prompt_text))
        
        if not choice_matches:
            print(f"   Warning: No choices found for participant {participant_idx}, skipping...")
            continue
        
        print(f"   Processing participant {participant_data.get('participant', participant_idx)}: {len(choice_matches)} choices")
        
        # Tokenize the full prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Single forward pass for the entire participant prompt
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Extract NLL for each choice token progressively
            for choice_idx, match in enumerate(choice_matches):
                choice_value = match.group(1)  # The number inside <<>>
                choice_start_pos = match.start() + 12  # Position after "You press <<"
                
                # Tokenize text up to the choice to find token position
                text_before_choice = prompt_text[:choice_start_pos]
                tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
                choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1
                
                # Tokenize just the choice content (the number)
                choice_tokens = tokenizer(choice_value, return_tensors="pt", add_special_tokens=False)
                choice_token_ids = choice_tokens['input_ids'][0]
                
                # Extract logits for predicting the choice tokens
                if choice_start_token_pos + len(choice_token_ids) <= len(logits):
                    choice_logits = logits[choice_start_token_pos:choice_start_token_pos + len(choice_token_ids)]
                    
                    # Compute NLL for each choice token
                    individual_token_nlls = []
                    for i, token_id in enumerate(choice_token_ids):
                        if i < len(choice_logits):
                            token_logits = choice_logits[i]
                            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                            token_nll = -log_probs[token_id].item()
                            individual_token_nlls.append(token_nll)
                    
                    # Average NLL for this choice
                    if individual_token_nlls:
                        choice_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                        all_individual_token_nlls.extend(individual_token_nlls)
                    else:
                        choice_nll = float('inf')
                else:
                    choice_nll = float('inf')
                    individual_token_nlls = []
                
                # Store simple trial result
                trial_result = {
                    'trial_index': len(per_trial_results),
                    'participant': participant_data.get('participant'),
                    'choice_index': choice_idx,
                    'ground_truth_choice': choice_value,
                    'trial_nll': choice_nll,
                    'has_history': choice_idx > 0,
                    'num_history_choices': choice_idx,
                    'dataset': condition_name
                }
                per_trial_results.append(trial_result)
    
    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(all_individual_token_nlls) / len(all_individual_token_nlls) if all_individual_token_nlls else float('inf')
    
    print(f"   ✅ Progressive NLL extraction complete")
    print(f"   🎯 Overall NLL: {overall_nll:.4f}")
    print(f"   🔍 Valid choices: {len(valid_trial_nlls)}/{len(per_trial_results)}")
    print(f"   📊 Total tokens evaluated: {len(all_individual_token_nlls)}")
    
    # Create basic results structure (simple like ruggeri format)
    basic_results = {
        'condition': condition_name,
        'nll': overall_nll,
        'num_samples': len(valid_trial_nlls),
        'total_choices_processed': len(per_trial_results),
        'total_participants_processed': len(dataset_list),
        'evaluation_method': 'Progressive NLL extraction from full participant prompts (wu2018generalisation_exp1)',
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create simple detailed results structure
    detailed_results = {
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_trial_nlls),
            'total_tokens_evaluated': len(all_individual_token_nlls),
            'overall_nll': overall_nll,
            'trials_with_history': len([r for r in per_trial_results if r.get('has_history', False)]),
            'trials_without_history': len([r for r in per_trial_results if not r.get('has_history', False)])
        },
        'collection_method': 'Progressive NLL extraction with single forward pass per participant (wu2018generalisation_exp1)',
        'condition': condition_name
    }
    
    return overall_nll, len(valid_trial_nlls), basic_results, detailed_results

def evaluate_with_progressive_nll_unified(model, tokenizer, dataset_list, condition_name, batch_size=4):
    """
    Unified progressive NLL extraction for all datasets.
    One forward pass per participant, extract NLL for each choice token progressively.
    
    Note: For context-free evaluation, we process participants individually because 
    each participant has a different behavioral sequence length and we need to 
    extract NLL progressively for each choice in their sequence.
    """
    print(f"🔍 Evaluating {condition_name} with unified progressive NLL extraction")
    print(f"   Processing {len(dataset_list)} participant prompts")
    print(f"   Method: One forward pass per participant (progressive extraction)")
    
    device = next(model.parameters()).device
    per_trial_results = []
    all_individual_token_nlls = []
    
    # Use tqdm for better progress tracking
    from tqdm import tqdm
    
    for participant_idx, participant_data in tqdm(enumerate(dataset_list), 
                                                  total=len(dataset_list),
                                                  desc="Processing participants",
                                                  unit="participant"):
        prompt_text = participant_data['text']
        
        # Extract choice patterns using a universal regex pattern
        import re
        # Try multiple choice patterns to cover different datasets
        choice_patterns = [
            r'You press <<(\d+)>>',  # wu2018, collsioo
            r'You choose <<(\w+)>>',  # hilbig, ruggeri 
            r'You select <<([^>]+)>>',  # general selection
            r'<<(\d+)>>',  # simple number choices
            r'<<([^>]+)>>'  # any choice in brackets
        ]
        
        choice_matches = []
        for pattern in choice_patterns:
            matches = list(re.finditer(pattern, prompt_text))
            if matches:
                choice_matches = matches
                break
        
        if not choice_matches:
            tqdm.write(f"   Warning: No choices found for participant {participant_idx}, skipping...")
            continue
        
        # Tokenize the full participant prompt once
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Single forward pass for the entire participant prompt
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Extract NLL for each choice token progressively
            for choice_idx, match in enumerate(choice_matches):
                choice_value = match.group(1)  # The content inside <<>>
                choice_start_pos = match.end() - len(choice_value) - 2  # Position of choice content
                
                # Tokenize text up to the choice to find token position
                text_before_choice = prompt_text[:choice_start_pos]
                tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
                choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1
                
                # Tokenize just the choice content
                choice_tokens = tokenizer(choice_value, return_tensors="pt", add_special_tokens=False)
                choice_token_ids = choice_tokens['input_ids'][0]
                
                # Extract logits for predicting the choice tokens
                if choice_start_token_pos + len(choice_token_ids) <= len(logits):
                    choice_logits = logits[choice_start_token_pos:choice_start_token_pos + len(choice_token_ids)]
                    
                    # Compute NLL for each choice token
                    individual_token_nlls = []
                    for i, token_id in enumerate(choice_token_ids):
                        if i < len(choice_logits):
                            token_logits = choice_logits[i]
                            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                            token_nll = -log_probs[token_id].item()
                            individual_token_nlls.append(token_nll)
                    
                    # Average NLL for this choice
                    if individual_token_nlls:
                        choice_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                        all_individual_token_nlls.extend(individual_token_nlls)
                    else:
                        choice_nll = float('inf')
                else:
                    choice_nll = float('inf')
                    individual_token_nlls = []
                
                # Store simple trial result
                trial_result = {
                    'trial_index': len(per_trial_results),
                    'participant': participant_data.get('participant'),
                    'choice_index': choice_idx,
                    'ground_truth_choice': choice_value,
                    'trial_nll': choice_nll,
                    'has_history': choice_idx > 0,
                    'num_history_choices': choice_idx,
                    'dataset': condition_name
                }
                per_trial_results.append(trial_result)
        
        # Update progress every few participants
        if (participant_idx + 1) % 5 == 0:
            recent_nlls = [r['trial_nll'] for r in per_trial_results[-50:] if r['trial_nll'] != float('inf')]
            avg_recent_nll = sum(recent_nlls) / len(recent_nlls) if recent_nlls else float('inf')
            tqdm.write(f"     Participant {participant_idx + 1}: {len(choice_matches)} choices, recent avg NLL: {avg_recent_nll:.3f}")
    
    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(all_individual_token_nlls) / len(all_individual_token_nlls) if all_individual_token_nlls else float('inf')
    
    print(f"   ✅ Progressive NLL extraction complete")
    print(f"   🎯 Overall NLL: {overall_nll:.4f}")
    print(f"   🔍 Valid choices: {len(valid_trial_nlls)}/{len(per_trial_results)}")
    print(f"   📊 Total tokens evaluated: {len(all_individual_token_nlls)}")
    
    # Create basic results structure (simple like ruggeri format)
    basic_results = {
        'condition': condition_name,
        'nll': overall_nll,
        'num_samples': len(valid_trial_nlls),
        'total_choices_processed': len(per_trial_results),
        'total_participants_processed': len(dataset_list),
        'evaluation_method': 'Unified progressive NLL extraction from full participant prompts',
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create simple detailed results structure
    detailed_results = {
        'per_trial_results': per_trial_results,
        'summary_statistics': {
            'total_trials': len(per_trial_results),
            'valid_trials': len(valid_trial_nlls),
            'total_tokens_evaluated': len(all_individual_token_nlls),
            'overall_nll': overall_nll,
            'trials_with_history': len([r for r in per_trial_results if r.get('has_history', False)]),
            'trials_without_history': len([r for r in per_trial_results if not r.get('has_history', False)])
        },
        'collection_method': 'Unified progressive NLL extraction with single forward pass per participant',
        'condition': condition_name
    }
    
    return overall_nll, len(valid_trial_nlls), basic_results, detailed_results

def create_history_only_prompts(original_prompts_file, output_file, dataset_name):
    """Create history-only prompts using dataset-specific functions"""
    print(f"📝 Creating history-only prompts from: {original_prompts_file}")
    print(f"   Using dataset-specific parsing for: {dataset_name}")
    
    # Route to appropriate dataset-specific function
    if dataset_name == "ruggeri2022globalizability":
        return create_history_only_prompts_ruggeri(original_prompts_file, output_file)
    elif dataset_name == "dubois2022value":
        return create_history_only_prompts_dubois(original_prompts_file, output_file)
    elif dataset_name == "wu2018generalisation_exp1":
        return create_history_only_prompts_wu2018(original_prompts_file, output_file)
    elif dataset_name == "collsioo2023MCPL_exp1":
        return create_history_only_prompts_collsioo(original_prompts_file, output_file)
    elif dataset_name == "collsioo2023MCPL_all":
        dataset_config = get_dataset_config(dataset_name)
        experiment_files = dataset_config['original_prompts_file']
        return create_history_only_prompts_collsioo_multi(experiment_files, output_file)
    elif dataset_name == "hilbig2014generalized":
        return create_history_only_prompts_hilbig(original_prompts_file, output_file)
    elif dataset_name == "hebart2023things":
        return create_history_only_prompts_hebart(original_prompts_file, output_file)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_dataset_config(dataset_name):
    """Get configuration for different datasets"""
    configs = {
        "ruggeri2022globalizability": {
            "original_prompts_file": "test_datasets/main_test_tasks/ruggeri2022globalizability_exp1.jsonl",
            "baseline_nll": 0.4382756948471069,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.6590430736541748,  # from all_data_baseline.csv
            "random_nll": 0.6931471805599453,    # ln(2) for binary choices
            "description": "Temporal decision making task",
            "choice_pattern": r'You press <<([QL])>>'
        },
        "dubois2022value": {
            "original_prompts_file": "test_datasets/generalization_tasks/dubois2022value.jsonl",
            "baseline_nll": None,  # Not available - dubois2022value is in generalization tasks
            "cognitive_nll": None,  # Not available - dubois2022value is in generalization tasks
            "random_nll": 0.6931471805599453,    # ln(2) for binary choices
            "description": "Sequential value-based decision making task",
            "choice_pattern": r'You press <<([^>]+)>>'
        },
        "wu2018generalisation_exp1": {
            "original_prompts_file": "test_datasets/main_test_tasks/wu2018generalisation_exp1.jsonl",
            "baseline_nll": 1.8321380615234373,  # From provided Centaur results
            "cognitive_nll": 2.792812824249268,  # From provided cognitive model results  
            "random_nll": 3.4011973816621555,   # ln(30) for 30-choice task (options 1-30)
            "description": "Multi-environment exploration task with 30 options",
            "choice_pattern": r'You press <<(\d+)>>'
        },
        "collsioo2023MCPL_exp1": {
            "original_prompts_file": "test_datasets/main_test_tasks/collsiöö2023MCPL_exp1.jsonl",
            "baseline_nll": 1.123242735862732,  # Original Centaur performance on collsiöö2023MCPL
            "cognitive_nll": 1.9236862861598658,  # Cognitive model baseline on collsiöö2023MCPL
            "random_nll": 2.1972245773362196,   # Random guess baseline on collsiöö2023MCPL
            "description": "MCPL experiment 1: Progressive behavioral learning with minimal context",
            "choice_pattern": r'You say that the Caldionine concentration is <<([^>]+)>>'
        },
        "collsioo2023MCPL_all": {
            "original_prompts_file": [
                "test_datasets/main_test_tasks/collsiöö2023MCPL_exp1.jsonl",
                "test_datasets/main_test_tasks/collsiöö2023MCPL_exp2.jsonl",
                "test_datasets/main_test_tasks/collsiöö2023MCPL_exp3.jsonl"
            ],
            "baseline_nll": 1.123242735862732,  # Original Centaur performance on collsiöö2023MCPL
            "cognitive_nll": 1.9236862861598658,  # Cognitive model baseline on collsiöö2023MCPL
            "random_nll": 2.1972245773362196,   # Random guess baseline on collsiöö2023MCPL
            "description": "MCPL all experiments: Progressive behavioral learning across exp1, exp2, exp3",
            "choice_pattern": r'You say that the Caldionine concentration is <<([^>]+)>>'
        },
        "hilbig2014generalized": {
            "original_prompts_file": "test_datasets/main_test_tasks/hilbig2014generalized_exp1.jsonl",
            "baseline_nll": 0.0618637911975383,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.1922362762758062,  # from all_data_baseline.csv
            "random_nll": 0.6931471805599453,   # ln(2) for binary choices (P vs G)
            "description": "Multi-attribute decision making with expert ratings - ultra-minimal context",
            "choice_pattern": r'You press <<([PG])>>'
        },
        "hebart2023things": {
            "original_prompts_file": "test_datasets/main_test_tasks/hebart2023things_exp1.jsonl",
            "baseline_nll": 0.7851982712745667,  # from all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
            "cognitive_nll": 0.8343300819396973,  # from all_data_baseline.csv
            "random_nll": 1.0986122886681098,   # ln(3) for 3-choice task (A, B, C)
            "description": "THINGS odd-one-out task with 3-choice trials - ultra-minimal context",
            "choice_pattern": r'You press <<([ABC])>>'
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    return configs[dataset_name]

def evaluate_centaur_on_prompts(model_name, prompts_file, condition_name, batch_size=4, skip_detailed_analysis=False):
    """Evaluate Centaur on a set of prompts using progressive NLL extraction for all datasets"""
    print(f"🔍 Evaluating {model_name} on {condition_name}")
    print(f"   Using prompts: {prompts_file}")
    print(f"   Batch size: {batch_size}")
    print(f"   Using progressive NLL extraction (one forward pass per participant)")
    
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
    
    # Use progressive NLL extraction for all datasets (simplified unified approach)
    nll, num_samples, basic_results, detailed_results = evaluate_with_progressive_nll_unified(
        model, tokenizer, dataset_list, condition_name, batch_size
    )
    
    return nll, num_samples, basic_results, detailed_results

def create_comparison_plot(history_only_nll, dataset_config, task_name, output_file):
    """Create comparison plot showing History-Only Centaur vs baselines"""
    print(f"📊 Creating comparison plot...")
    
    # Build comparison data, filtering out None values
    comparison_data = [
        {'Model': 'History-Only Centaur\n(Context-Free)', 'NLL': history_only_nll, 'Type': 'Our Approach'}
    ]
    
    # Add baselines only if available
    if dataset_config['baseline_nll'] is not None:
        comparison_data.append({
            'Model': 'Original Centaur\n(Full Context)', 
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
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(7.08661, 3.5))
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Color scheme
    base_colors = ['#ff7f0e', '#69005f', '#cbc9e2']  # Orange (history-only), Purple, Light Purple
    colors = base_colors[:len(comp_df)]
    
    bars = ax.bar(range(len(comp_df)), comp_df['NLL'], color=colors, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Negative log-likelihood', fontsize=10)
    ax.set_title(f'Model Comparison: NLL on {task_name.replace("_", " ").title()}\n(Context-Free vs Full Context Evaluation)', fontsize=11)
    ax.set_xticks(range(len(comp_df)))
    ax.set_xticklabels(comp_df['Model'], rotation=0, ha='center', fontsize=9)
    
    # Add random baseline line
    random_nll = dataset_config['random_nll']
    ax.axhline(y=random_nll, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random guessing')
    ax.text(len(comp_df) - 0.1, random_nll + 0.05, 'Random guessing', 
           fontsize=9, color='red', horizontalalignment='right', fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add note
    note_text = 'Note: History-Only evaluation uses behavioral patterns without task-specific context'
    plt.figtext(0.02, 0.02, note_text, fontsize=7, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    png_file = output_file
    pdf_file = output_file.with_suffix('.pdf')
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Plot saved successfully")
    
    return comparison_data

def analyze_results(results_dict, dataset_config):
    """Analyze the results to understand what drives Centaur's performance"""
    print(f"\n🔍 Analysis: What drives Centaur's performance?")
    print(f"=" * 60)
    
    # Get results
    original_nll = results_dict.get('Original (Full Context)', {}).get('nll')
    history_nll = results_dict.get('History-Only', {}).get('nll')
    
    # Get baselines from config
    cognitive_nll = dataset_config.get('cognitive_nll')
    random_nll = dataset_config.get('random_nll')
    
    print(f"📊 Performance Summary:")
    if original_nll:
        print(f"   Original (Full Context):  {original_nll:.4f}")
    if history_nll:
        print(f"   Context-Free (History-Only): {history_nll:.4f}")
    if cognitive_nll:
        print(f"   Cognitive Models:        {cognitive_nll:.4f}")
    if random_nll:
        print(f"   Random Baseline:         {random_nll:.4f}")
    
    # Key comparisons
    print(f"\n🎯 Key Comparisons:")
    
    if original_nll and history_nll:
        context_effect = history_nll - original_nll
        print(f"   Context Dependency: {context_effect:.4f}")
        if context_effect > 0:
            print(f"   → Context-free performs WORSE (relies on task context)")
        else:
            print(f"   → Context-free performs BETTER (surprising!)")
    
    if history_nll and cognitive_nll:
        cognitive_gap = history_nll - cognitive_nll
        print(f"   vs Cognitive Models: {cognitive_gap:.4f}")
        if cognitive_gap > 0:
            print(f"   → Cognitive models outperform context-free Centaur")
        else:
            print(f"   → Context-free Centaur outperforms cognitive models")
    
    return {
        'context_effect': context_effect if original_nll and history_nll else None,
        'cognitive_gap': cognitive_gap if history_nll and cognitive_nll else None,
    }

def main():
    """Main evaluation pipeline - follows structure of evaluate_zero_shot_centaur.py"""
    parser = argparse.ArgumentParser(description='History-Only Centaur Evaluation')
    parser.add_argument('--task', type=str, 
                       choices=['ruggeri2022globalizability', 'dubois2022value', 'wu2018generalisation_exp1', 'collsioo2023MCPL_exp1', 'collsioo2023MCPL_all', 'hilbig2014generalized', 'hebart2023things'],
                       default='ruggeri2022globalizability',
                       help='Dataset to evaluate on')
    parser.add_argument('--model', '--model-name', dest='model_name', type=str, 
                       default='marcelbinz/Llama-3.1-Centaur-70B-adapter',
                       help='Model to evaluate')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-detailed-analysis', action='store_true',
                       help='Skip detailed per-trial analysis to avoid hanging (saves basic results only)')
    parser.add_argument('--run-original', action='store_true',
                       help='Run original full-context evaluation (default: use existing baseline)')
    
    args = parser.parse_args()
    
    print("=== History-Only Centaur Evaluation ===")
    print("Testing: Task Context vs Behavioral History")
    print("Question: Does Centaur capture cognitive patterns or superficial behavioral patterns?")
    print()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.task)
    
    # Setup
    model_name = args.model_name
    task_name = args.task
    
    # File paths from config
    original_prompts_file = dataset_config['original_prompts_file']
    
    # Create context_free_eval directory
    eval_results_dir = Path("context_free_eval")
    eval_results_dir.mkdir(exist_ok=True)
    
    # Output files
    history_only_file = eval_results_dir / f"{task_name}_history_only_prompts.jsonl"
    
    print(f"📋 Configuration:")
    print(f"   Dataset: {task_name}")
    print(f"   Description: {dataset_config['description']}")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Original prompts: {original_prompts_file}")
    print(f"   History-only prompts: {history_only_file}")
    print(f"   Original Centaur baseline: {'Will evaluate' if args.run_original else 'Use existing baseline'}")
    
    # Check if original prompts file(s) exist
    if isinstance(original_prompts_file, list):
        # Multiple files (e.g., collsioo2023MCPL_all)
        missing_files = []
        for file_path in original_prompts_file:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        if missing_files:
            print(f"❌ Error: Original prompts files not found: {missing_files}")
            return None
    else:
        # Single file
        if not os.path.exists(original_prompts_file):
            print(f"❌ Error: Original prompts file not found: {original_prompts_file}")
            return None
    
    # Step 1: Create history-only prompts
    print(f"\n=== Step 1: Create History-Only Prompts ===")
    print(f"   Using dataset-specific parsing for: {task_name}")
    
    # Create history-only prompts
    print("📝 Creating history-only prompts...")
    history_only_prompts = create_history_only_prompts(original_prompts_file, history_only_file, task_name)
    
    # Step 2: Evaluate different conditions
    print(f"\n=== Step 2: Evaluate Different Conditions ===")
    
    results_dict = {}
    
    # Evaluate history-only (our main focus)
    print("\n🔍 Evaluating History-Only condition...")
    # Use efficient evaluation for large datasets, detailed analysis otherwise
    history_nll, history_samples, history_basic, history_detailed = evaluate_centaur_on_prompts(
        args.model_name, history_only_file, "History-Only", args.batch_size, args.skip_detailed_analysis
    )
    results_dict['History-Only'] = {
        'nll': history_nll, 
        'samples': history_samples,
        'basic_results': history_basic,
        'detailed_results': history_detailed
    }
    

    
    # Original full context (default: use existing baseline)
    if args.run_original:
        print("\n🔍 Evaluating Original (Full Context) condition...")
        original_nll, original_samples, original_basic, original_detailed = evaluate_centaur_on_prompts(
            args.model_name, original_prompts_file, "Original (Full Context)", args.batch_size, args.skip_detailed_analysis
        )
        results_dict['Original (Full Context)'] = {
            'nll': original_nll, 
            'samples': original_samples,
            'basic_results': original_basic,
            'detailed_results': original_detailed
        }
    else:
        # Use known baseline from config (default behavior)
        baseline_nll = dataset_config['baseline_nll']
        if baseline_nll:
            results_dict['Original (Full Context)'] = {'nll': baseline_nll, 'samples': 'from_baseline'}
            print(f"\n📊 Using existing Original Centaur baseline: {baseline_nll:.4f}")
        else:
            print(f"\n⚠️  No baseline available for {task_name}")
    
    # Step 3: Create comparison plot
    print(f"\n=== Step 3: Create Comparison Plot ===")
    try:
        plot_file = eval_results_dir / f"{task_name}_context_free_comparison.png"
        comparison_data = create_comparison_plot(history_nll, dataset_config, task_name, plot_file)
    except Exception as e:
        print(f"   ❌ Error creating comparison plot: {e}")
        print(f"   ⚠️  Continuing without plot...")
        comparison_data = [{'Model': 'History-Only Centaur\n(Context-Free)', 'NLL': history_nll, 'Type': 'Our Approach'}]
    
    # Step 4: Analyze results
    print(f"\n=== Step 4: Analyze Results ===")
    analysis = analyze_results(results_dict, dataset_config)
    
    # Step 5: Save comprehensive results
    print(f"\n=== Step 5: Save Comprehensive Results ===")
    
    # Get baseline values from config (same as original script)
    original_centaur_nll = dataset_config['baseline_nll']
    cognitive_models_nll = dataset_config['cognitive_nll']
    random_nll = dataset_config['random_nll']
    
    comprehensive_results = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'script_version': 'context_free_centaur_evaluation_v2.0',
            'dataset_name': task_name,
            'dataset_description': dataset_config['description'],
            'model_name': model_name,
            'original_prompts_file': original_prompts_file,
            'history_only_prompts_file': str(history_only_file),
            'method': 'Context-free evaluation with behavioral history only',
            'difference': 'Removes task-specific context, keeps behavioral patterns',
            'parsing_method': f'Dataset-specific parsing for {task_name}',
            'purpose': 'Test whether Centaur captures cognitive patterns vs superficial behavioral patterns'
        },
        'prompt_data': {
            'original_prompts_count': len(history_only_prompts),
            'history_only_prompts_count': len(history_only_prompts),
            'history_only_prompts': [],  # Skip redundant prompt data - using unified progressive NLL extraction for all tasks
            'prompt_creation_method': f'Dataset-specific parsing for {task_name}',
            'note': 'Prompts not saved - using unified progressive NLL extraction for all datasets to save only trial-level NLL results'
        },
        'evaluation_results': {
            'context_free_centaur': {
                'nll': history_nll,
                'basic_results': history_basic,
                'detailed_results': history_detailed,
                'trial_specific_data': {
                    'description': 'Individual NLL for each choice prediction',
                    'total_trials': len(history_detailed['per_trial_results']) if history_detailed and 'per_trial_results' in history_detailed else 0,
                    'sample_trials': history_detailed['per_trial_results'][:10] if history_detailed and 'per_trial_results' in history_detailed else [],
                    'summary_stats': history_detailed['summary_statistics'] if history_detailed and 'summary_statistics' in history_detailed else None
                }
            },
            'published_baselines': {
                'original_centaur_full_context': {
                    'nll': original_centaur_nll, 
                    'available': original_centaur_nll is not None
                },
                'cognitive_models': {
                    'nll': cognitive_models_nll,
                    'available': cognitive_models_nll is not None
                },
                'random': {
                    'nll': random_nll,
                    'available': True
                }
            }
        },
        'comparison_analysis': {
            'context_free_vs_original': {
                'context_free_nll': history_nll,
                'original_nll': original_centaur_nll,
                'difference': history_nll - original_centaur_nll if original_centaur_nll else None,
                'context_free_worse': history_nll > original_centaur_nll if original_centaur_nll else None,
                'interpretation': ('Context-free performs worse than full context' if original_centaur_nll and history_nll > original_centaur_nll 
                                 else 'Context-free performs better than full context' if original_centaur_nll 
                                 else 'Original baseline not available')
            },
            'context_free_vs_cognitive': {
                'context_free_nll': history_nll,
                'cognitive_nll': cognitive_models_nll,
                'difference': history_nll - cognitive_models_nll if cognitive_models_nll else None,
                'cognitive_better': cognitive_models_nll < history_nll if cognitive_models_nll else None,
                'interpretation': ('Cognitive models outperform context-free Centaur' if cognitive_models_nll and cognitive_models_nll < history_nll 
                                 else 'Context-free Centaur outperforms cognitive models' if cognitive_models_nll 
                                 else 'Cognitive models baseline not available')
            },
            'comparison_data': comparison_data
        },
        'visualization': {
            'plot_files': [f'context_free_eval/{task_name}_context_free_comparison.png', f'context_free_eval/{task_name}_context_free_comparison.pdf'],
            'color_scheme': {
                'context_free_centaur': '#ff7f0e',
                'original_centaur': '#69005f',
                'cognitive_models': '#cbc9e2',
                'note': 'Context-free evaluation removes task-specific information'
            },
            'styling': 'Original paper styling (nature theme from scienceplots)',
            'approach': 'Context-free evaluation using behavioral patterns only'
        }
    }
    
    # Add original results if evaluated
    if 'Original (Full Context)' in results_dict and 'basic_results' in results_dict['Original (Full Context)']:
        comprehensive_results['evaluation_results']['original_full_context'] = {
            'nll': results_dict['Original (Full Context)']['nll'],
            'basic_results': results_dict['Original (Full Context)']['basic_results'],
            'detailed_results': results_dict['Original (Full Context)'].get('detailed_results')
        }
    
    results_file = eval_results_dir / f"{task_name}_context_free_results.json"
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
        simplified_file = eval_results_dir / f"{task_name}_simplified_context_free_results.json"
        with open(simplified_file, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"   ✅ Saved simplified results to {simplified_file}")
        results_file = simplified_file
    
    print(f"✅ Evaluation Complete!")
    print(f"📊 Results Summary:")
    print(f"   Dataset: {task_name}")
    print(f"   Context-Free Centaur NLL: {history_nll:.4f}")
    if original_centaur_nll:
        print(f"   Original Centaur NLL: {original_centaur_nll:.4f}")
    else:
        print(f"   Original Centaur NLL: Not available")
    if cognitive_models_nll:
        print(f"   Cognitive Models NLL: {cognitive_models_nll:.4f}")
    else:
        print(f"   Cognitive Models NLL: Not available")
    print(f"   Random NLL: {random_nll:.4f}")
    print(f"📈 Plots saved as: context_free_eval/{task_name}_context_free_comparison.png & context_free_eval/{task_name}_context_free_comparison.pdf")
    print(f"   Plot shows: Context-Free Centaur, Original Centaur, Cognitive Models, Random line")
    print(f"📄 Comprehensive results saved as: {results_file}")
    print(f"   Data includes: All baselines and detailed per-trial analysis")
    
    # Show what complete results were saved
    print(f"\n💾 Complete Results Saved:")
    print(f"   📊 Overall metrics: NLL, evaluation config, metadata")
    
    if history_detailed and 'per_trial_results' in history_detailed:
        trial_count = len(history_detailed['per_trial_results'])
        print(f"   🎯 Per-trial NLL results: {trial_count} individual choice predictions")
        print(f"   📝 Each trial includes: trial_nll, prompt_text, ground_truth, choice_index, has_history")
        print(f"   📈 Summary statistics: mean/std/min/max/median trial NLL")
        print(f"   🔍 Sample data: First 10 trials included for inspection")
        
        # Show example trial data structure
        if history_detailed['per_trial_results']:
            sample_trial = history_detailed['per_trial_results'][0]
            print(f"   📋 Example trial keys: {list(sample_trial.keys())}")
            print(f"   🎲 Sample trial NLL: {sample_trial.get('trial_nll', 'N/A'):.4f}")
    else:
        print(f"   📝 Basic results only (detailed analysis skipped)")
    
    print(f"   🔍 Full prompts: All {len(history_only_prompts)} context-free prompts with parsing details")
    print(f"   📋 This enables detailed post-hoc analysis of behavioral pattern capture")
    
    # Analysis
    print(f"\n🔍 Analysis:")
    if original_centaur_nll:
        if history_nll > original_centaur_nll:
            print(f"   ✅ Context-free Centaur performs WORSE than full context ({history_nll:.3f} vs {original_centaur_nll:.3f})")
            print(f"   ✅ This suggests Centaur relies on task-specific context!")
        else:
            print(f"   ⚠️  Context-free Centaur performs BETTER than full context ({history_nll:.3f} vs {original_centaur_nll:.3f})")
            print(f"   ⚠️  This needs further investigation")
    else:
        print(f"   ℹ️  Original Centaur baseline not available for {task_name}")
    
    if cognitive_models_nll:
        if history_nll > cognitive_models_nll:
            print(f"   ✅ Cognitive models outperform context-free Centaur ({cognitive_models_nll:.3f} vs {history_nll:.3f})")
            print(f"   ✅ This supports cognitive models' superiority in behavioral pattern capture!")
        else:
            print(f"   ⚠️  Context-free Centaur outperforms cognitive models ({history_nll:.3f} vs {cognitive_models_nll:.3f})")
            print(f"   ⚠️  This suggests Centaur captures deeper behavioral patterns")
    else:
        print(f"   ℹ️  Cognitive models baseline not available for {task_name}")
    
    # Key insight about behavioral pattern capture
    print(f"\n💡 Key Insight:")
    if original_centaur_nll and cognitive_models_nll:
        context_gap = abs(history_nll - original_centaur_nll)
        cognitive_gap = abs(history_nll - cognitive_models_nll)
        
        if context_gap > 0.1:
            print(f"   ⚠️  Large performance gap ({context_gap:.3f}) between context-free and full context")
            print(f"   → Centaur heavily relies on task-specific information")
            if cognitive_gap < context_gap:
                print(f"   → But closer to cognitive models ({cognitive_gap:.3f}) suggests some behavioral pattern capture")
            else:
                print(f"   → Risk of superficial pattern matching rather than cognitive modeling")
        else:
            print(f"   ✅ Small performance gap ({context_gap:.3f}) between context-free and full context")
            print(f"   → Centaur captures behavioral patterns beyond task-specific context")
            print(f"   → Evidence for cognitive-level pattern learning")
    
    return comprehensive_results

if __name__ == "__main__":
    main() 