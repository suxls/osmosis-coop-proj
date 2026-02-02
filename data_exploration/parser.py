import os
import sys

import pandas as pd

SYSTEM_PROMPT = """
You are a Lean4 auto-formalization engine.

## Task
Given an informal theorem and proof in natural language, translate it into executable Lean4 code.

## Input Format
Theorem: [Natural language theorem statement with LaTeX notation]
Proof: [Natural language proof ending with "This completes the proof"]

## Output Format
```lean4
[imports]
[namespace/open declarations]
[variable declarations]
theorem [name] [statement] := by
  /- [proof comment] -/
  [tactics]
```

## Example
Input:
Theorem: For types α and β with relations r and s, if s is a well-order on β, then for any two initial segment embeddings f, g : r ≼i s and any element a ∈ α, it holds that f(a) = g(a).
Proof: Since s is a well-order, the type r ≼i s is a subsingleton. Thus f = g, so f(a) = g(a). This completes the proof.

Output:
```lean4
import Mathlib.Order.InitialSeg
open InitialSeg

variable {α : Type*} {β : Type*} {r : α → α → Prop} {s : β → β → Prop}

theorem InitialSeg.eq_x [IsWellOrder β s] (f g : r ≼i s) (a) : f a = g a := by
  /- Since r ≼i s is a subsingleton, any two embeddings are equal. -/
  rw [Subsingleton.elim f g]
```

## Requirements
- Code must compile in Lean4
- All goals must be closed
- Use idiomatic Lean4 style
- Preserve informal proof structure in comments
- Include only necessary imports

Respond with valid Lean4 code only.
"""

def load_herald_df(parquet_path: str = "herald.parquet") -> pd.DataFrame:
    """Load the Herald parquet file."""
    return pd.read_parquet(parquet_path)


def generate_user_prompt(row: pd.Series) -> str:
    """
    Return string in the form:
    Theorem: {informal_theorem}
    Proof: {informal_proof}
    """
    return f"Theorem:\n{row['informal_theorem']}\nProof:\n{row['informal_proof']}"


# def modify_header(header) -> str:
#     """
#     TODO: connect to ai, to modify the header to make sure there is no invalid imports or declarations and rewrite if needed 
#     """
#     return header

def generate_ground_truth(row: pd.Series) -> str:
    """
    Return string concatenating header and formal_proof for the entry.
    """
    return f"{row['header']}\n{row['formal_proof']}"


def _row_to_pair(row: pd.Series) -> dict:
    """Build a single NL-FL pair from a dataframe row."""
    return {
        "user_prompt": generate_user_prompt(row),
        "system_prompt": SYSTEM_PROMPT,
        "ground_truth": generate_ground_truth(row),
    }


def generate_NL_FL_pairs(
    n: int,
    parquet_path: str = "herald.parquet",
    random_state: int | None = None,
) -> list[dict]:
    """
    Generate n NL-FL pairs by sampling rows from the given parquet file.

    user_prompt: from ["informal_theorem"] and ["informal_proof"] columns.
    ground_truth: from ["header"] and ["formal_proof"] columns.

    Returns a list of dicts with keys: user_prompt, system_prompt, ground_truth.
    """
    df = load_herald_df(parquet_path)
    if n > len(df):
        raise ValueError(f"Requested n={n} rows but parquet has only {len(df)} rows.")
    sample = df.sample(n=n, random_state=random_state)
    return [_row_to_pair(row) for _, row in sample.iterrows()]


def write_parquet(output_path: str, pairs: list[dict]) -> None:
    """Write NL-FL pairs to a Parquet file."""
    df = pd.DataFrame(pairs)
    df.to_parquet(output_path, index=False)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Wrote {len(pairs):,} rows to {output_path} ({size_kb:.2f} KB)")


def main() -> int:
    """Prompt for filename and number of rows, then write output.parquet in the same directory."""
    print("NL-FL pairs: sample from herald.parquet and write user_prompt / system_prompt / ground_truth.\n")

    try:
        filename = input("Output filename (e.g. train): ").strip()
        if not filename:
            print("Error: filename is required")
            return 1
        n_str = input("Number of rows: ").strip()
        n = int(n_str)
        if n <= 0:
            print("Error: number of rows must be positive")
            return 1
    except ValueError:
        print("Error: number of rows must be an integer")
        return 1

    if not filename.endswith(".parquet"):
        filename += ".parquet"

    # Same path = current working directory
    output_path = os.path.join(os.getcwd(), filename)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_parquet = os.path.join(os.getcwd(), "herald.parquet")
    if not os.path.exists(source_parquet):
        source_parquet = os.path.join(script_dir, "herald.parquet")
    if not os.path.exists(source_parquet):
        print("Error: source parquet not found (tried cwd and script directory)")
        return 1

    print(f"Sampling {n:,} rows from {source_parquet}...")
    try:
        pairs = generate_NL_FL_pairs(n, parquet_path=source_parquet)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    write_parquet(output_path, pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
