import re
import unicodedata
from collections import Counter

from osmosis_ai import osmosis_reward


# --- Lean4 hygiene pipeline (quotes/dashes only; keep Lean unicode like α, β) ---
_ASCII_QUOTES = (
    ("\u2018", "'"),   # LEFT SINGLE QUOTE
    ("\u2019", "'"),   # RIGHT SINGLE QUOTE
    ("\u201c", '"'),   # LEFT DOUBLE QUOTE
    ("\u201d", '"'),   # RIGHT DOUBLE QUOTE
    ("\u201a", "'"),   # SINGLE LOW-9 QUOTE
    ("\u201b", "'"),   # SINGLE HIGH-REVERSED-9 QUOTE
    ("\u201e", '"'),   # DOUBLE LOW-9 QUOTE
    ("\u201f", '"'),   # DOUBLE HIGH-REVERSED-9 QUOTE
    ("\u2039", "'"),   # SINGLE LEFT-POINTING ANGLE QUOTE
    ("\u203a", "'"),   # SINGLE RIGHT-POINTING ANGLE QUOTE
)
_ASCII_DASHES = (
    ("\u2010", "-"),   # HYPHEN
    ("\u2011", "-"),   # NON-BREAKING HYPHEN
    ("\u2012", "-"),   # FIGURE DASH
    ("\u2013", "-"),   # EN DASH
    ("\u2014", "-"),   # EM DASH
    ("\u2015", "-"),   # HORIZONTAL BAR
    ("\u2212", "-"),   # MINUS SIGN
)


def _lean_hygiene_pipeline(s: str, rename_identifiers: bool = False) -> str:
    """
    Hygiene pipeline for LLM-produced Lean4 code:

      LLM output → Unicode NFC → ASCII sanitize (quotes, dashes) → [optional identifier renaming]
    """
    if not s:
        return s
    # 1. Unicode normalization (NFC)
    out = unicodedata.normalize("NFC", s)
    # 2. ASCII sanitize (quotes, dashes)
    for old, new in _ASCII_QUOTES + _ASCII_DASHES:
        out = out.replace(old, new)
    # 3. Optional identifier renaming (no-op by default; extend for custom rules)
    if rename_identifiers:
        # Placeholder: e.g. normalize identifier unicode to preferred form
        pass
    return out

def extract_solution(solution_str):
    solution = re.search(r'####\s*([-+]?\d*\.?\d+)', solution_str)
    if(not solution or solution is None):
        return None
    final_solution = solution.group(1)
    return final_solution


def _normalize_lean(s: str) -> str:
    """Normalize Lean4 snippet for equivalence: strip and collapse whitespace."""
    return " ".join(s.strip().split())


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (bag-of-tokens with multiplicity). Tokens = whitespace-split."""
    pred_tokens = pred.strip().split()
    gold_tokens = gold.strip().split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if not pred_tokens and not gold_tokens else 0.0
    pred_cnt = Counter(pred_tokens)
    gold_cnt = Counter(gold_tokens)
    common = sum((pred_cnt & gold_cnt).values())
    tp = common
    pred_total = sum(pred_cnt.values())
    gold_total = sum(gold_cnt.values())
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

# @osmosis_reward
def lean4_compile_reward(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    rename_identifiers: bool = False,
    **kwargs,
) -> float:
    """
    RL reward for Lean4: semantic equivalence + token F1 + length penalty.

    solution_str is run through the hygiene pipeline before comparison:
      Unicode NFC → ASCII sanitize (quotes, dashes) → [optional identifier renaming]

    reward = 1.0 * semantic_validity + 0.2 * token_f1(pred, gold) - 0.01 * len(pred.split())

    - semantic_validity: 1.0 if pred and gold are equivalent after normalizing whitespace, else 0.0.
    - token_f1: bag-of-tokens F1 between solution and ground_truth.
    - length penalty: discourages verbose solutions.
    """
    pred = _lean_hygiene_pipeline(solution_str.strip(), rename_identifiers=rename_identifiers)
    gold = _lean_hygiene_pipeline(ground_truth.strip(), rename_identifiers=rename_identifiers)
    semantic_validity = 1.0 if _normalize_lean(pred) == _normalize_lean(gold) else 0.0
    token_f1_score = _token_f1(pred, gold)
    # length_penalty = 0.01 * len(pred.split())
    reward = 1.0 * semantic_validity + 0.2 * token_f1_score # - length_penalty
    print(f"reward: {reward}")
    print(f"semantic_validity: {semantic_validity}")
    print(f"token_f1_score: {token_f1_score}")
    # print(f"length_penalty: {length_penalty}")
    return min(reward, 1.0)



solution_str = "import Mathlib.Logic.Equiv.Set import Mathlib.Order.RelIso.Set import Mathlib.Order.WellFounded import Mathlib.Order.InitialSeg open InitialSeg variable {α : Type*} {β : Type*} {γ : Type*} {r : α → α → Prop} {s : β → β → Prop} {t : γ → γ → Prop} open Function theorem InitialSeg.eq [IsWellOrder β s] (f g : r ≼i s) (a) : f a = g a := by rw [Subsingleton.elim f g]"
ground_truth = "import Mathlib.Logic.Equiv.Set import Mathlib.Order.RelIso.Set import Mathlib.Order.WellFounded import Mathlib.Order.InitialSeg open InitialSeg variable {α : Type*} {β : Type*} {γ : Type*} {r : α → α → Prop} {s : β → β → Prop} {t : γ → γ → Prop} open Function theorem InitialSeg.eq [IsWellOrder β s] (f g : r ≼i s) (a) : f a = g a := by rw [Subsingleton.elim f g]"
print(lean4_compile_reward(solution_str, ground_truth))