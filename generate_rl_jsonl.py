import json
from pathlib import Path
import pandas as pd
import argparse
import numpy as np


def build_cover_story(total_trials: int, label_left: str = "U", label_right: str = "P") -> str:
    # Align exactly with predictive_rl_centaur_Sabrina.py's build_slot_prompt preface
    preface = (
        f"In this task, you have to repeatedly choose between two slot machines labeled {label_left} and {label_right}.\n"
        "You can choose a slot machine by pressing its corresponding key."
        "When you select one of the machines, you will win 1 or 0 points."
        "Your goal is to choose the slot machines that will give you the most points."
        "You will receive feedback about the outcome after making a choice.\n"
        "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
        f"You will play 1 game in total, consisting of {total_trials} trials."
        f" Game 1:"
    )
    return preface


def convert_csv_to_jsonl(input_csv: Path, output_jsonl: Path, experiment_tag: str = "predictive_rl/exp1.csv") -> None:
    """
    Convert predictive RL CSV data into Centaur-style JSONL.

    Supported inputs:
    - Summary CSV with columns [model_id, overall_nll, prompt], where prompt is already
      a complete Centaur prompt.
    - Trial-level CSV with columns [trial, choice, reward, cumulative_reward, model_id].
    """
    df = pd.read_csv(input_csv)

    if {"model_id", "prompt"}.issubset(df.columns):
        participants = []
        for _, row in df.iterrows():
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty prompt for model_id={row.get('model_id')}")
            participants.append({
                "text": prompt,
                "experiment": experiment_tag,
                "participant": str(row["model_id"]),
            })

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for rec in participants:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Converted summary CSV with prompt column.")
        print(f"Wrote {len(participants)} participants to {output_jsonl}")
        return

    # Basic validation
    required_cols = {"trial", "choice", "reward", "model_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    # Map integer choices to letters (0->U, 1->P)
    choice_map = {0: "U", 1: "P"}
    if df["choice"].dtype != object:
        df["choice_letter"] = df["choice"].map(choice_map)
    else:
        # If already letters, keep as-is
        df["choice_letter"] = df["choice"]

    # Ensure trials are sorted within participant
    if "trial" in df.columns:
        df = df.sort_values(["model_id", "trial"])

    # Group by participant (model_id)
    participants = []
    for model_id, g in df.groupby("model_id"):
        g = g.reset_index(drop=True)
        total_trials = len(g)
        cover = build_cover_story(total_trials, "U", "P")

        # Build per-trial lines
        # Format: "You press <<U>> and get 1 points."
        lines = [cover, ""]
        for _, row in g.iterrows():
            choice_letter = str(row["choice_letter"]).strip()
            reward = int(row["reward"])
            lines.append(f"You press <<{choice_letter}>> and get {reward} points.")

        # Add Sabrina-style open cue for the next choice (unfinished marker)
        lines.append("You press <<")

        text = "\n".join(lines)
        participants.append({
            "text": text,
            "experiment": experiment_tag,
            "participant": str(model_id),
        })

    # Write JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in participants:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(participants)} participants to {output_jsonl}")


# ---------------- WCST converter ----------------

def build_wcst_cover_story() -> str:
    lines = [
        "You will see a stimulus card and must choose which of four key cards it matches. Cards can match by one of three categories: color, form, or number. The matching category changes from time to time. After each choice, you will receive a feedback stimulus:",
        '- "REPEAT" means you used the correct category and should keep using it.',
        '- "SWITCH" means you used the wrong category and should try a different one.',
        "",
        "The four key cards are always:",
        "A: one red triangle",
        "B: two green stars",
        "C: three yellow crosses",
        "D: four blue balls",
        "",
        "Each stimulus card shares at most one property (color, form, or number) with any one key card.",
        "Your task is to use the feedbacks to figure out the correct temporary category to apply and respond accordingly pressing key 'A' or 'B' or 'C' or 'D'.",
    ]
    return "\n".join(lines)


def convert_wcst_npy_to_jsonl(input_npy: Path, output_jsonl: Path, experiment_tag: str = "wcst/predictive.npy") -> None:
    """
    Convert json_wcst_dataset.npy into Centaur-style JSONL.
    - One JSONL row per subject (participant = subject_id)
    - Text contains the WCST cover story and per-trial lines mirroring predictive_centaur_70B_wcst.py
    """
    data = np.load(input_npy, allow_pickle=True)

    LETTER2NUM = {"A": 1, "B": 2, "C": 3, "D": 4}
    NUM2LETTER = {v: k for k, v in LETTER2NUM.items()}

    # Group steps by subject_id while preserving order
    by_subject = {}
    for step in data:
        sid = int(step["subject_id"])
        by_subject.setdefault(sid, []).append(step)

    participants = []
    cover = build_wcst_cover_story()

    for sid, steps in by_subject.items():
        lines = [cover, ""]
        for step in steps:
            stim = step["stimulus"]  # expects [color, number, form] as used in the reference code
            # Match the same ordering used in predictive_centaur_70B_wcst.py: {2} {0} {1}
            lines.append(f"You see the following stimulus card: {stim[2]} {stim[0]} {stim[1]}. You press key <<",)

            user_key = int(step["user_key"])  # 1..4
            letter = NUM2LETTER[user_key]

            # key_cards is indexed by numeric key; replicate finish_step_prompt formatting
            kc = step["key_cards"][user_key]
            key_card_desc = f"{kc[2]} {kc[0]} {kc[1]}"
            feedback = "REPEAT" if user_key == int(step["ground_key"]) else "SWITCH"

            # Close the open '<<' with the chosen letter and add feedback line
            lines[-1] = lines[-1] + f"{letter}>> ({key_card_desc})."
            lines.append(f"You get the following feedback stimulus: {feedback}.")

        text = "\n".join(lines)
        participants.append({
            "text": text,
            "experiment": experiment_tag,
            "participant": str(sid),
        })

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in participants:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(participants)} WCST participants to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate Centaur-style JSONL prompts from behavioral data.")

    # RL CSV mode (default)
    parser.add_argument("--input-csv", type=Path, default=Path("test_data_rl.csv"),
                        help="Input CSV path with columns [trial, choice, reward, cumulative_reward, model_id].")
    parser.add_argument("--output-jsonl", type=Path, default=Path("datasets/main_test_tasks/predictive_rl_exp1.jsonl"),
                        help="Output JSONL path to write Centaur-style prompts (RL).")
    parser.add_argument("--experiment-tag", type=str, default="predictive_rl/exp1.csv",
                        help="Value to populate the 'experiment' field (RL).")

    # WCST NPY mode
    parser.add_argument("--wcst", action="store_true",
                        help="If set, convert WCST npy (json_wcst_dataset.npy) to Centaur-style JSONL instead of RL CSV.")
    parser.add_argument("--input-npy", type=Path, default=Path("json_wcst_dataset.npy"),
                        help="Input WCST .npy path (e.g., json_wcst_dataset.npy).")
    parser.add_argument("--output-jsonl-wcst", type=Path, default=Path("datasets/main_test_tasks/wcst_predictive.jsonl"),
                        help="Output JSONL path to write Centaur-style prompts (WCST).")
    parser.add_argument("--experiment-tag-wcst", type=str, default="wcst/predictive.npy",
                        help="Value to populate the 'experiment' field (WCST).")

    args = parser.parse_args()

    if args.wcst:
        convert_wcst_npy_to_jsonl(args.input_npy, args.output_jsonl_wcst, experiment_tag=args.experiment_tag_wcst)
    else:
        convert_csv_to_jsonl(args.input_csv, args.output_jsonl, experiment_tag=args.experiment_tag)


if __name__ == "__main__":
    main()


