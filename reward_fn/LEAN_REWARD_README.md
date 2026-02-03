# Lean Proof Reward Function Documentation

## Overview

The `lean_proof_reward` function is a reward function designed for RL training loops that work with Lean 4 mathematical proofs. It verifies whether a given Lean proof string is valid by checking it against the Lean 4 kernel.

## How It Works

### Architecture

```
solution_str (string)
    ↓
Write to temp.lean file in lean_env/
    ↓
Run: lake env lean temp.lean
    ↓
Check exit code & stderr
    ↓
Return 1.0 (valid) or 0.0 (invalid)
    ↓
Clean up temp file
```

### Step-by-Step Process

1. **Input**: Receives a `solution_str` containing Lean 4 proof code as a string

2. **Environment Setup**:
   - Locates the `lean_env/` directory (sibling to `reward_fn/`)
   - This directory contains a configured Lean 4 v4.11.0 environment with Lake

3. **Temporary File Creation**:
   - Creates a temporary `.lean` file inside the `lean_env/` directory
   - Writes the `solution_str` content to this file

4. **Verification**:
   - Executes `lake env lean <temp_file>` from within the `lean_env/` directory
   - `lake env` ensures the correct Lean version (v4.11.0) is used
   - The Lean kernel checks the proof for correctness

5. **Result Evaluation**:
   - If return code is 0 and no errors in stderr → **Reward: 1.0**
   - If return code is non-zero or errors present → **Reward: 0.0**
   - If timeout (30 seconds) is exceeded → **Reward: 0.0**

6. **Cleanup**:
   - Removes the temporary `.lean` file

## Function Signature

```python
@osmosis_reward
def lean_proof_reward(solution_str: str, ground_truth: str = None) -> float:
    """
    Reward function for Lean 4 proofs.

    Args:
        solution_str: String containing the Lean proof code
        ground_truth: Optional (not used for kernel verification)

    Returns:
        float: 1.0 if proof is valid, 0.0 otherwise
    """
```

## Dependencies

- **Lean 4**: v4.11.0 (configured via `lean_env/lean-toolchain`)
- **Lake**: Lean's build tool (must be installed and in PATH)
- **osmosis_ai**: For the `@osmosis_reward` decorator
- **Python**: 3.12+ (as specified in pyproject.toml)

## Environment Structure

```
osmosis-coop-proj/
├── reward_fn/
│   ├── lean_reward.py          # The reward function
│   └── LEAN_REWARD_README.md   # This documentation
└── lean_env/                    # Lean 4 environment
    ├── lean-toolchain           # Specifies v4.11.0
    ├── lakefile.toml            # Lake configuration
    └── ...                      # Other Lean project files
```

## Usage Examples

See `lean_reward_examples.py` for detailed test cases and usage patterns.

### Basic Usage

```python
from reward_fn.lean_reward import lean_proof_reward

# Example 1: Valid proof
proof = "theorem add_comm : 2 + 3 = 5 := rfl"
reward = lean_proof_reward(proof)
# Returns: 1.0

# Example 2: Invalid proof
invalid = "theorem wrong : 1 + 1 = 3 := rfl"
reward = lean_proof_reward(invalid)
# Returns: 0.0
```

## Error Handling

The function handles various error cases gracefully:

| Error Type | Behavior | Return Value |
|------------|----------|--------------|
| Invalid syntax | Caught by Lean compiler | 0.0 |
| False theorem | Rejected by kernel | 0.0 |
| Timeout (>30s) | Process killed | 0.0 |
| Missing lean_env | Exception caught | 0.0 |
| File I/O errors | Exception caught | 0.0 |

## Performance Considerations

- **Timeout**: Each verification has a 30-second timeout
- **Temp Files**: Created in `lean_env/` and cleaned up immediately
- **Process Overhead**: Each call spawns a new Lean process via subprocess

## Integration with RL Training

This reward function is designed to be used in RL training loops where:
- The agent generates Lean proof strings
- The reward function provides immediate feedback (1.0 or 0.0)
- The agent learns to generate valid Lean proofs

```python
# Pseudocode for RL training loop
for episode in training_episodes:
    proof_string = agent.generate_proof(problem)
    reward = lean_proof_reward(proof_string)
    agent.update_policy(reward)
```

## Troubleshooting

### Issue: Returns 0.0 for all inputs

**Solution**: Check that:
1. `lean_env/` directory exists in project root
2. Lean 4 and Lake are installed: `which lean` and `which lake`
3. The Lean environment was built: `cd lean_env && lake build`

### Issue: Timeout errors

**Solution**:
- Check if the proof is too complex or has infinite loops
- Increase timeout in `lean_reward.py` line 51 if needed

### Issue: ImportError for osmosis_ai

**Solution**: Install dependencies:
```bash
source .venv/bin/activate
pip install osmosis-ai
```

## Version Information

- **Lean**: v4.11.0 (leanprover/lean4:v4.11.0)
- **Python**: >=3.12
- **osmosis_ai**: >=0.2.2
