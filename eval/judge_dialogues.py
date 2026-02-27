import argparse
import json
import logging
import os
import re
import numpy as np
from tqdm import tqdm
from transcribe_dialogues import transcribe_dial
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from utils import get_wav_pairs, load_asr_model
from training.inference_utils import (
    average_results,
    save_judgements_to_json,
    save_summary_to_json,
)


def get_pipe(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device=0, seed=42):
    if model_id != "meta-llama/Meta-Llama-3.1-8B-Instruct":
        raise NotImplementedError(f"Model {model_id} not supported as judge.")
    
    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto",
    )
    # HF text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        do_sample=True,  # Enable sampling for variation across seeds
        temperature=0.7,  # Add temperature for controlled randomness
    )
    return pipe


def merge_dialogues(dialogue_1, dialogue_2):
    def extract_speaker(text, speaker):
        pattern = rf"{speaker}:\s*(.*)"
        return re.findall(pattern, text)

    speaker1_from_d1 = extract_speaker(dialogue_1, "Speaker 1")
    speaker2_from_d2 = extract_speaker(dialogue_2, "Speaker 2")

    merged_dialogue = []
    for line in speaker1_from_d1:
        merged_dialogue.append(f"Speaker 1: {line}")
    for line in speaker2_from_d2:
        merged_dialogue.append(f"Speaker 2: {line}")

    final_output = "\n".join(merged_dialogue)
    return final_output


def judge_dialogues(pipe, transcription):
    system_prompt = (
        "You are an expert evaluator of dialogues. "
        "You only respond with a single digit from 1 to 5 based on the evaluation criteria."
    )
    user_prompt = (
        "Evaluate the following dialogue transcript on these criteria:\n"
        "1. Coherence: Are the responses relevant and logical?\n"
        "2. Engagement: Is the dialogue interesting?\n"
        "3. Fluency: Is the language natural and correct?\n"
        "4. Creativity/Originality: Does it show creativity in responses?\n\n"
        "If the dialogue stops abruptly or seems cut off at the end, do NOT penalize it for that.\n\n"
        "Score the dialogue strictly between 1 and 5 (1=Very poor, 5=Excellent). "
        "Output ONLY the score as a single number, no text or punctuation.\n\n"
        f"Dialogue:\n{transcription}\nScore:"
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"
    outputs = pipe(prompt)
    text = outputs[0]["generated_text"][len(prompt) :].strip()
    match = re.search(r"\b[1-5]\b", text)
    score = match.group(0) if match else "Invalid"
    return text, score


def judge_narratives(pipe, transcription, narrative):
    system_prompt = (
        "You are an expert evaluator of dialogues. "
        "You only respond with a single digit from 1 to 5 based on the evaluation criteria."
    )
    user_prompt = (
        "Evaluate how well the following dialogue fits the given narrative.\n\n"
        "Criteria:\n"
        "1. Relevance: Does the dialogue clearly reflect the situation or topic described in the narrative?\n"
        "2. Consistency: Are the characters, events, and tone in the dialogue consistent with the narrative?\n"
        "3. Faithfulness: Does the dialogue avoid introducing contradictions or unrelated content?\n\n"
        "- Do NOT judge fluency or engagement — only topical/narrative alignment.\n"
        "- Score strictly between 1 and 5 (1 = Not related at all, 5 = Perfectly fits the narrative).\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"Dialogue:\n{transcription}\n\n"
        "Score:"
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"
    outputs = pipe(prompt)
    text = outputs[0]["generated_text"][len(prompt) :].strip()
    match = re.search(r"\b[1-5]\b", text)
    score = match.group(0) if match else "Invalid"
    return text, score


def check_starting_speaker(transcript, s1_starts):
    speaker_pattern = r"^(Speaker \d+)"
    for line in transcript.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(speaker_pattern, line)
        if match:
            first_speaker = match.group(1)
            actual_s1_starts = first_speaker == "Speaker 1"
            return int(actual_s1_starts == s1_starts)
        else:
            print("OH NO, no transcript")
    return 0


def speaker_starts(prompt: str) -> bool:
    match = re.search(
        r"-\s*starts\s+the\s+dialogue\s*:\s*(True|False)", prompt, re.IGNORECASE
    )
    if match:
        return match.group(1).lower() == "true"
    return False


def extract_narrative(prompt: str) -> str | None:
    """
    Extracts the narrative section from a dialogue-generation prompt.
    Returns the narrative string, or None if not found.
    """
    match = re.search(r"(?s)Narrative:\s*(.*?)\n\s*Your behaviors:", prompt)
    if match:
        return match.group(1).strip(" -\n ")
    return None


def get_instructions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [
        (int(entry["soda_index"]), entry["instruction_s1"], entry["narrative"])
        for entry in data
    ]

    narratives = {}
    s1_speaks_first = {}

    for soda_index, p, narrative in prompts:
        s1_speaks_first[soda_index] = speaker_starts(p)
        narratives[soda_index] = narrative

    return narratives, s1_speaks_first


def get_soda_index(paths):
    path = paths[0]
    match = re.search(r"soda_index_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def run_single_evaluation(args, logger, seed, dialogues, narratives, s1_speaks_first):
    """Run evaluation with a single seed"""
    logger.info(f"Running evaluation with seed {seed}")
    
    pipeline = get_pipe(args.judge, seed=seed)
    
    # Judge dialogues
    judgements_dialogues = [
        judge_dialogues(pipeline, d[1])
        for d in tqdm(dialogues, desc=f"Judging dialogues (seed {seed})")
    ]
    
    # Judge narratives
    judgements_narratives = [
        judge_narratives(pipeline, d[1], narratives[d[0]])
        for d in tqdm(dialogues, desc=f"Judging narratives (seed {seed})")
    ]
    
    # Calculate results
    results = {}
    for eval_type, judgments in [("narrative", judgements_narratives), 
                                   ("dialogue", judgements_dialogues)]:
        avg_score, invalid_pct, invalid_count, total = average_results(judgments)
        results[eval_type] = {
            "avg_score": avg_score,
            "invalid_pct": invalid_pct,
            "invalid_count": invalid_count,
            "total": total,
            "judgements": judgments
        }
    
    return results


def main(args, logger):
    logger.info(f"Using judge: {args.judge}")
    logger.info(f"Reading generated output from: {args.wavs_dir}")
    logger.info(f"Running with {args.num_seeds} different seeds: {args.seeds}")

    # Load data (only once)
    narratives, s1_speaks_first = get_instructions(args.instruction_file)
    wavs_pairs = get_wav_pairs(args.wavs_dir)
    asr_pipe = load_asr_model()

    # Transcribe dialogues (only once)
    logger.info("Transcribing dialogues...")
    dialogues = [
        (get_soda_index(sample), transcribe_dial(sample, asr_pipe, no_timestamps=True))
        for sample in tqdm(wavs_pairs, total=len(wavs_pairs))
    ]

    # Check starting speakers (only once, doesn't depend on LLM seed)
    starting_speakers_results = [
        check_starting_speaker(d[1], s1_speaks_first[d[0]]) for d in tqdm(dialogues)
    ]
    correct_first_speakers_percentage = (
        sum(starting_speakers_results) / len(starting_speakers_results) * 100
    )


    logger.info(f"Correct first speakers: {correct_first_speakers_percentage:.1f}%")

    # Run evaluation with multiple seeds
    all_seed_results = []
    for seed in args.seeds:
        seed_results = run_single_evaluation(
            args, logger, seed, dialogues, narratives, s1_speaks_first
        )
        all_seed_results.append(seed_results)
        
        # Save individual seed results
        for eval_type in ["narrative", "dialogue"]:
            score_output_file = os.path.join(
                args.output_dir, f"output_judge_results_{eval_type}_seed{seed}.json"
            )
            save_judgements_to_json(
                dialogues, 
                seed_results[eval_type]["judgements"], 
                score_output_file
            )

    # Aggregate results across seeds
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS ACROSS ALL SEEDS")
    logger.info("="*80)
    
    summary_dict = {}
    
    for eval_type in ["narrative", "dialogue"]:
        scores = [r[eval_type]["avg_score"] for r in all_seed_results]
        invalid_pcts = [r[eval_type]["invalid_pct"] for r in all_seed_results]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        mean_invalid = np.mean(invalid_pcts)
        std_invalid = np.std(invalid_pcts, ddof=1) if len(invalid_pcts) > 1 else 0.0
        
        logger.info(f"\n[{eval_type.upper()}]")
        logger.info(f"  Average score: {mean_score:.3f} ± {std_score:.3f}")
        logger.info(f"  Individual seed scores: {[f'{s:.3f}' for s in scores]}")
        logger.info(f"  Invalid responses: {mean_invalid:.2f}% ± {std_invalid:.2f}%")
        
        summary_dict[eval_type] = {
            "mean_score": mean_score,
            "std_score": std_score,
            "individual_scores": scores,
            "mean_invalid_pct": mean_invalid,
            "std_invalid_pct": std_invalid,
            "individual_invalid_pcts": invalid_pcts,
        }
    
    summary_dict["first_speaker_correct_pct"] = correct_first_speakers_percentage
    summary_dict["num_seeds"] = args.num_seeds
    summary_dict["seeds_used"] = args.seeds
    
    # Save final summary
    summary_output_file = os.path.join(
        args.output_dir, "eval_results_dialogue_judge_multi_seed.json"
    )
    save_summary_to_json(args.judge, summary_dict, summary_output_file)
    
    logger.info("\n" + "="*80)
    logger.info("Everything done! Goodbye!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process generated output and evaluate with a judge."
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model-id of the judge",
    )
    parser.add_argument(
        "--wavs_dir",
        type=str,
        required=True,
        help="Path to directory containing WAV files",
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        required=True,
        help="Path to directory containing instructions (and general outputs)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the result"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of different seeds to run (default: 3)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="List of seeds to use (default: [42, 123, 456])",
    )

    args = parser.parse_args()

    # Validate seeds argument
    if len(args.seeds) != args.num_seeds:
        parser.error(
            f"Number of seeds provided ({len(args.seeds)}) must match --num_seeds ({args.num_seeds})"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "llm_judge_seeds.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(args, logger)