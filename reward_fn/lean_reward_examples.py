"""
Simple test script for lean_proof_reward function.

Usage:
1. Edit the INPUT_STRING variable below with your Lean proof
2. Run: python reward_fn/lean_reward_examples.py
3. See the score (1.0 or 0.0) and failure reason if applicable
"""

import os
import tempfile
import subprocess
from pathlib import Path

# ============================================================================
# EDIT THIS: Put your Lean proof string here
# ============================================================================
INPUT_STRING = """
import Mathlib open scoped ENNReal NNReal Nat open MeasureTheory Real Set Filter Topology theorem poissonPMFRealSum_extracted (r : ℝ≥0) : HasSum (fun n => ProbabilityTheory.poissonPMFReal r n) 1 := sorry

"""

# ============================================================================
# Test function (includes detailed error reporting)
# ============================================================================

def test_lean_proof_with_details(solution_str: str):
    """
    Test a Lean proof and return detailed results.

    Returns:
        tuple: (score, error_message)
            score: 1.0 if valid, 0.0 if invalid
            error_message: None if valid, error details if invalid
    """
    try:
        # Get the path to the lean_env directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        lean_env_dir = project_root / "lean_env"

        # Verify lean_env directory exists
        if not lean_env_dir.exists():
            return 0.0, f"Lean environment not found at {lean_env_dir}"

        # Create a temporary .lean file in the lean_env directory
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            dir=lean_env_dir,
            delete=False
        ) as temp_file:
            temp_file.write(solution_str)
            temp_file_path = temp_file.name

        try:
            # Use lake env lean to verify the proof
            result = subprocess.run(
                ['lake', 'env', 'lean', temp_file_path],
                cwd=lean_env_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Check if the proof is valid
            if result.returncode == 0:
                stderr_lower = result.stderr.lower()
                if 'error' not in stderr_lower or len(result.stderr.strip()) == 0:
                    return 1.0, None

            # Collect error information
            error_msg = "Lean compiler errors:\n"
            if result.stderr:
                error_msg += f"\nSTDERR:\n{result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT:\n{result.stdout}"
            error_msg += f"\nReturn code: {result.returncode}"

            return 0.0, error_msg

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except subprocess.TimeoutExpired:
        return 0.0, "Timeout: Proof took longer than 30 seconds to verify"
    except Exception as e:
        return 0.0, f"Exception occurred: {str(e)}"


# ============================================================================
# Run the test
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LEAN PROOF REWARD TEST")
    print("="*70)

    print("\nInput String:")
    print("-"*70)
    print(INPUT_STRING)
    print("-"*70)

    print("\nRunning Lean kernel check...")
    score, error = test_lean_proof_with_details(INPUT_STRING)

    print("\n" + "="*70)
    print(f"RESULT: {score}")
    print("="*70)

    if score == 1.0:
        print("\n✓ SUCCESS: Proof is valid!")
        print("The Lean kernel accepted this proof.")
    else:
        print("\n✗ FAILURE: Proof is invalid")
        print("\nReason for failure:")
        print("-"*70)
        print(error)
        print("-"*70)

    print("\n" + "="*70)
