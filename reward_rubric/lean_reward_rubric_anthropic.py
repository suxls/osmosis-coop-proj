from osmosis_ai import (
    evaluate_rubric,
    osmosis_rubric,
)
import os

# make life easier by hardcoding the rubric, score range and model info
RUBRIC = """
You are a Lean4 statement similarity evaluator.

You are given two strings:
1. A predicted Lean4 statement (model output)
2. A ground-truth Lean4 statement (expected answer)

Your task is to output a single numeric reward between 0 and 1.

- If the predicted string is empty, malformed, or not a valid Lean4 statement,
  output 0 immediately.
- If the predicted statement does not parse or elaborate in Lean4,
  output 0 immediately.
- Output 1 if the predicted statement is mathematically and definitionally
  identical to the ground-truth statement (up to alpha-renaming, formatting,
  and definitional equality).
- Otherwise, output a real number strictly between 0 and 1 representing how
  close the predicted statement is to the ground truth.

Output exactly one number between 0 and 1.
"""


SCORE_MIN = 0.0
SCORE_MAX = 1.0
PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-5-20250929"
API_KEY = os.getenv("ANTHROPIC_API_KEY")

@osmosis_rubric
def compute_rubric_score_anthropic(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs
) -> float:
    """
    Delegate rubric scoring to a hosted model while keeping @osmosis_rubric validation.
    """
    model_info = {"provider": PROVIDER, "model": MODEL, "api_key": API_KEY}
    prompt_metadata = extra_info.get("metadata")

    result = evaluate_rubric(
        rubric=RUBRIC,
        solution_str=solution_str,
        model_info=model_info,
        ground_truth=ground_truth,
        metadata=prompt_metadata,
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
        return_details=False,
    )

    return float(result)