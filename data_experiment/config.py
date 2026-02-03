import pandas as pd

SYSTEM_PROMPT = """
You are a Lean4 auto-formalization engine.

## Task
Given a statement of a math problem,translate it into formal language Lean4 code.
Do not write any proof step for this theorem or try to solve this problem , you should focus on the translation ONLY.
Simply use "sorry" as a place holder of the detailed proof if necessary. For
example , 1+1=2 is translated into:
lean example : 1+1=2 := sorry
"""


# Cache the workbook in-memory so we don't re-read it per row.
_WORKBOOK_DF: pd.DataFrame | None = None


def _get_workbook_df() -> pd.DataFrame:
    global _WORKBOOK_DF
    if _WORKBOOK_DF is None:
        _WORKBOOK_DF = pd.read_parquet("workbook.parquet")
    return _WORKBOOK_DF


def generate_lean_example() -> dict:
    '''
    Generate a single Lean4 auto-formalization training example.
    '''
    df = _get_workbook_df()
    row = df.sample(n=1).iloc[0]

    nl_statement = row["natural_language_statement"]
    formal_statement = row["formal_statement"]

    return {
        # Ensure plain Python strings (not pandas Series) for Arrow/JSON serialization.
        "user_prompt": str(nl_statement),
        "system_prompt": SYSTEM_PROMPT.strip(),
        "ground_truth": str(formal_statement),
    }
