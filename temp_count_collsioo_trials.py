import json
import re
import argparse
import pandas as pd
from pathlib import Path


def count_trials_in_prompt(text: str) -> int:
    # Define a trial as any occurrence of the choice statement
    return len(re.findall(r'You say that the Caldionine concentration is <<[^>]+>>', text))


def count_choice_tokens_in_prompt(text: str) -> int:
    # Count occurrences of <<...>> regardless of surrounding context
    return len(re.findall(r'<<[^>]+>>', text))


def count_choice_tokens_in_trials(text: str) -> int:
    # Count <<...>> only in actual trial choice statements
    return len(re.findall(r'You say that the Caldionine concentration is <<[^>]+>>', text))


def count_experiment_trials(file_path: Path) -> int:
    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return 0

    total_trials = 0
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            total_trials += count_trials_in_prompt(text)
    return total_trials


def count_experiment_choice_tokens(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    total = 0
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            total += count_choice_tokens_in_prompt(text)
    return total


def count_experiment_choice_tokens_breakdown(file_path: Path) -> tuple[int, int, int]:
    """Return (total_tokens, tokens_in_trials, tokens_outside_trials) for one experiment file."""
    if not file_path.exists():
        return (0, 0, 0)
    total = 0
    in_trials = 0
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            total += count_choice_tokens_in_prompt(text)
            in_trials += count_choice_tokens_in_trials(text)
    return (total, in_trials, total - in_trials)


def main():
    parser = argparse.ArgumentParser(description='Count trials for collsioo2023MCPL experiments (exp1, exp2, exp3). Also compute per-experiment mean NLL from CSV for context_free evaluation.')
    parser.add_argument('--base', type=str, default=r'E:\reanalyzing_centaur\test_datasets\main_test_tasks',
                        help='Base directory containing collsiöö2023MCPL_exp{1,2,3}.jsonl')
    parser.add_argument('--nll_csv', type=str, default='trial_level_nlls_long_format.csv',
                        help='Path to trial_level_nlls_long_format.csv to compute per-experiment mean NLLs')
    args = parser.parse_args()

    base = Path(args.base)
    files = [
        ('exp1', base / 'collsiöö2023MCPL_exp1.jsonl'),
        ('exp2', base / 'collsiöö2023MCPL_exp2.jsonl'),
        ('exp3', base / 'collsiöö2023MCPL_exp3.jsonl'),
    ]

    print('Counting trials for collsioo2023MCPL (all experiments)')
    print(f'Using base directory: {base}')
    print('Note: If files are missing, download from OSF (placeholder link).')

    per_exp = {}
    per_exp_choice_tokens = {}
    per_exp_choice_tokens_in_trials = {}
    per_exp_choice_tokens_outside_trials = {}
    for tag, path in files:
        n = count_experiment_trials(path)
        per_exp[tag] = n
        print(f'  {tag}: {n} trials')

        total_c, in_trials_c, outside_c = count_experiment_choice_tokens_breakdown(path)
        per_exp_choice_tokens[tag] = total_c
        per_exp_choice_tokens_in_trials[tag] = in_trials_c
        per_exp_choice_tokens_outside_trials[tag] = outside_c
        print(f'  {tag}: {total_c} occurrences of <<>> tokens (in trials: {in_trials_c}, outside: {outside_c})')

    total = sum(per_exp.values())
    print(f'Total trials across exp1+exp2+exp3: {total}')
    total_choice_tokens = sum(per_exp_choice_tokens.values())
    total_choice_tokens_in_trials = sum(per_exp_choice_tokens_in_trials.values())
    total_choice_tokens_outside = sum(per_exp_choice_tokens_outside_trials.values())
    print(f'Total <<>> occurrences across exp1+exp2+exp3: {total_choice_tokens} (in trials: {total_choice_tokens_in_trials}, outside: {total_choice_tokens_outside})')

    # Compute average NLL per experiment (context_free) if CSV available
    csv_path = Path(args.nll_csv)
    if not csv_path.exists():
        print(f"NLL CSV not found at: {csv_path}. Skipping per-experiment NLL averages.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV {csv_path}: {e}")
        return

    df_cf = df[(df['dataset'] == 'collsioo2023MCPL_all') & (df['evaluation_type'] == 'context_free')].copy()
    if df_cf.empty:
        print('No context_free rows found for collsioo2023MCPL_all in the CSV.')
        return

    # Cast types
    df_cf['trial_index'] = df_cf['trial_index'].astype(int)
    df_cf['nll'] = df_cf['nll'].astype(float)

    # Determine boundaries based on counts
    exp1_count = per_exp.get('exp1', 0)
    exp2_count = per_exp.get('exp2', 0)
    exp3_count = per_exp.get('exp3', 0)

    b1_start, b1_end = 0, exp1_count - 1
    b2_start, b2_end = exp1_count, exp1_count + exp2_count - 1
    b3_start, b3_end = exp1_count + exp2_count, exp1_count + exp2_count + exp3_count - 1

    def mean_in_range(start: int, end: int) -> float:
        if end < start:
            return float('nan')
        sub = df_cf[(df_cf['trial_index'] >= start) & (df_cf['trial_index'] <= end)]
        return float(sub['nll'].mean()) if not sub.empty else float('nan')

    exp1_mean = mean_in_range(b1_start, b1_end)
    exp2_mean = mean_in_range(b2_start, b2_end)
    exp3_mean = mean_in_range(b3_start, b3_end)

    print('\nAverage context_free NLL by experiment (from CSV):')
    print(f"  exp1 [{b1_start}-{b1_end}]: {exp1_mean:.6f}")
    print(f"  exp2 [{b2_start}-{b2_end}]: {exp2_mean:.6f}")
    print(f"  exp3 [{b3_start}-{b3_end}]: {exp3_mean:.6f}")


if __name__ == '__main__':
    main()


