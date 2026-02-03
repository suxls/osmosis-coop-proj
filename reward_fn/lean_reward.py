import os
import tempfile
import subprocess
from pathlib import Path
from osmosis_ai import osmosis_reward


def _lean_verify_with_reason(solution_str: str) -> tuple[float, str]:
    """
    Run Lean kernel check on solution_str.
    Returns (reward, reason): (1.0, success_msg) or (0.0, failure_output).
    """
    try:
        reward_fn_dir = Path(__file__).parent
        project_root = reward_fn_dir.parent
        lean_env_dir = project_root / "lean_env"

        if not lean_env_dir.exists():
            return 0.0, f"Lean environment not found at {lean_env_dir}"

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            dir=lean_env_dir,
            delete=False
        ) as temp_file:
            temp_file.write(solution_str)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                ['lake', 'env', 'lean', temp_file_path],
                cwd=lean_env_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                stderr_lower = (result.stderr or "").lower()
                if 'error' not in stderr_lower or len((result.stderr or "").strip()) == 0:
                    return 1.0, "Proof is valid."

            # Failure: prefer stderr, then stdout, then returncode
            out = (result.stderr or "").strip() or (result.stdout or "").strip()
            if not out:
                out = f"Lean exited with code {result.returncode} (no output)."
            return 0.0, out

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except subprocess.TimeoutExpired:
        return 0.0, "Lean verification timed out (30s)."
    except Exception as e:
        return 0.0, f"Error verifying proof: {e}"


@osmosis_reward
def lean_proof_reward(solution_str: str, ground_truth: str = None):
    """
    Reward function for Lean 4 proofs.
    Returns 1.0 if the proof passes the Lean kernel check, 0.0 otherwise.

    Args:
        solution_str: String containing the Lean proof code
        ground_truth: Optional ground truth (not used for kernel verification)

    Returns:
        float: 1.0 if proof is valid, 0.0 otherwise
    """
    reward, _ = _lean_verify_with_reason(solution_str)
    return reward


def lean_proof_reward_with_reason(solution_str: str, ground_truth: str = None) -> tuple[float, str]:
    """
    Same as lean_proof_reward but returns (reward, reason) so callers can show why it failed.
    reason is e.g. "Proof is valid." on success, or Lean stderr / error message on failure.
    """
    return _lean_verify_with_reason(solution_str)

