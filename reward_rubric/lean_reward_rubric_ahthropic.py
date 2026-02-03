from osmosis_ai import (
    evaluate_rubric,
    osmosis_rubric,
)
import os

# make life easier by hardcoding the rubric, score range and model info
RUBRIC = """
You are a Lean4 proof evaluator.

You are given two strings:
1. A predicted Lean4 file (model output)
2. A ground-truth Lean4 file (expected solution)

Your task is to determine whether the two files are mathematically and formally equivalent.

Equivalence criteria:
- Both files must be valid Lean4 code.
- The predicted file must not be empty, truncated, malformed, or in an invalid format.
- The predicted file must successfully compile in Lean4 (warnings are allowed; errors are not).
- The predicted file must define the same theorems, definitions, or statements as the ground truth.
- All hypotheses, assumptions, types, and conclusions must match exactly in meaning.
- Differences in formatting, whitespace, comments, variable names, proof style, or proof strategy are allowed.
- The use of different tactics or intermediate lemmas is allowed as long as the final statements are equivalent.
- Any missing theorem, extra incorrect theorem, weakened/strengthened condition, or incorrect conclusion makes the files NOT equivalent.
Output:
- Return `true` if and only if the predicted Lean4 file is valid, compiles, and is mathematically equivalent to the ground truth.
- Otherwise, return `false`.
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