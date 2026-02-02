from typing import Dict, Tuple, Optional
import subprocess
import tempfile
import os
from dataclasses import dataclass

@dataclass
class RewardComponents:
    """Individual components of the reward function"""
    syntactic_correctness: float  # Does it parse?
    type_correctness: float       # Does it typecheck?
    semantic_equivalence: float   # Does it prove the same thing?
    structural_similarity: float  # Similar proof structure?
    efficiency: float             # Proof complexity/length
    
    def total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total reward"""
        if weights is None:
            weights = {
                'syntactic_correctness': 0.15,
                'type_correctness': 0.25,
                'semantic_equivalence': 0.40,
                'structural_similarity': 0.10,
                'efficiency': 0.10
            }
        
        return (
            weights['syntactic_correctness'] * self.syntactic_correctness +
            weights['type_correctness'] * self.type_correctness +
            weights['semantic_equivalence'] * self.semantic_equivalence +
            weights['structural_similarity'] * self.structural_similarity +
            weights['efficiency'] * self.efficiency
        )


class Lean4RewardFunction:
    """Reward function for evaluating Lean4 code equivalence"""
    
    def __init__(self, lean_path: str = "lean"):
        self.lean_path = lean_path
    
    def evaluate(self, solution_str: str, ground_truth: str) -> Tuple[float, RewardComponents]:
        """
        Main evaluation function that returns total reward and components
        
        Args:
            solution_str: LLM-generated Lean4 code
            ground_truth: Reference correct Lean4 code
            
        Returns:
            (total_reward, reward_components)
        """
        components = RewardComponents(
            syntactic_correctness=0.0,
            type_correctness=0.0,
            semantic_equivalence=0.0,
            structural_similarity=0.0,
            efficiency=0.0
        )
        
        # 1. Syntactic Correctness (0-1)
        components.syntactic_correctness = self._check_syntax(solution_str)
        
        if components.syntactic_correctness == 0:
            # If syntax is wrong, return early with zero reward
            return 0.0, components
        
        # 2. Type Correctness (0-1)
        components.type_correctness = self._check_types(solution_str, ground_truth)
        
        if components.type_correctness == 0:
            # If types don't match, limited reward possible
            return components.total_reward(), components
        
        # 3. Semantic Equivalence (0-1)
        components.semantic_equivalence = self._check_semantic_equivalence(
            solution_str, ground_truth
        )
        
        # 4. Structural Similarity (0-1)
        components.structural_similarity = self._compute_structural_similarity(
            solution_str, ground_truth
        )
        
        # 5. Efficiency Score (0-1)
        components.efficiency = self._compute_efficiency(solution_str, ground_truth)
        
        total_reward = components.total_reward()
        return total_reward, components
    
    def _check_syntax(self, code: str) -> float:
        """Check if code is syntactically valid Lean4"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                [self.lean_path, temp_path, '--stdin'],
                capture_output=True,
                timeout=5
            )
            
            os.unlink(temp_path)
            
            # Parse errors indicate syntax issues
            if b"error:" in result.stderr or result.returncode != 0:
                # Check if it's just a syntax error vs other issues
                if b"unexpected" in result.stderr or b"expected" in result.stderr:
                    return 0.0
                # Might be other errors (semantic), so give partial credit
                return 0.5
            
            return 1.0
            
        except Exception as e:
            return 0.0
    
    def _check_types(self, solution: str, ground_truth: str) -> float:
        """
        Check if solution has the same type signature as ground truth
        Returns 1.0 if types match, 0.5 if partially match, 0.0 otherwise
        """
        solution_type = self._extract_type_signature(solution)
        truth_type = self._extract_type_signature(ground_truth)
        
        if solution_type is None:
            return 0.0
        
        if solution_type == truth_type:
            return 1.0
        
        # Check for α-equivalence (renaming of variables)
        if self._are_alpha_equivalent(solution_type, truth_type):
            return 1.0
        
        # Partial credit for similar but not identical types
        similarity = self._type_similarity(solution_type, truth_type)
        return max(0.0, min(1.0, similarity))
    
    def _check_semantic_equivalence(self, solution: str, ground_truth: str) -> float:
        """
        Check if solution proves the same theorem as ground truth
        This is the most critical component
        """
        # Strategy 1: Direct verification - does the solution prove the goal?
        proves_goal = self._verify_proof(solution)
        if not proves_goal:
            return 0.0
        
        # Strategy 2: Cross-verification - can we prove each with the other?
        # Create a test that uses solution to prove ground_truth's statement
        cross_verify_score = self._cross_verify(solution, ground_truth)
        
        # Strategy 3: Check if they prove logically equivalent statements
        logical_equiv = self._check_logical_equivalence(solution, ground_truth)
        
        # Combine scores
        return max(cross_verify_score, logical_equiv)
    
    def _verify_proof(self, code: str) -> bool:
        """Verify that a proof is valid"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True,
                timeout=10
            )
            
            os.unlink(temp_path)
            
            # No errors means proof is valid
            return result.returncode == 0 and b"error:" not in result.stderr
            
        except Exception:
            return False
    
    def _cross_verify(self, solution: str, ground_truth: str) -> float:
        """
        Try to use solution's proof to verify ground truth's statement
        This checks if they prove the same thing
        """
        # Extract theorem statement from ground_truth
        truth_statement = self._extract_theorem_statement(ground_truth)
        solution_proof = self._extract_proof_term(solution)
        
        if truth_statement is None or solution_proof is None:
            return 0.0
        
        # Create a test file that uses solution's proof for truth's statement
        test_code = f"""
{solution}

-- Verification that solution proves the same statement
{truth_statement} := {solution_proof}
"""
        
        if self._verify_proof(test_code):
            return 1.0
        
        # Try the reverse
        solution_statement = self._extract_theorem_statement(solution)
        truth_proof = self._extract_proof_term(ground_truth)
        
        if solution_statement is None or truth_proof is None:
            return 0.5  # Partial credit for passing one direction
        
        reverse_test = f"""
{ground_truth}

-- Reverse verification
{solution_statement} := {truth_proof}
"""
        
        if self._verify_proof(reverse_test):
            return 1.0
        
        return 0.5  # One direction worked
    
    def _check_logical_equivalence(self, solution: str, ground_truth: str) -> float:
        """Check if two proofs establish logically equivalent facts"""
        # This would involve more sophisticated analysis
        # For now, simplified version
        sol_statement = self._extract_theorem_statement(solution)
        truth_statement = self._extract_theorem_statement(ground_truth)
        
        if sol_statement == truth_statement:
            return 1.0
        
        # Could check for iff relationship, definitional equality, etc.
        return 0.0
    
    def _compute_structural_similarity(self, solution: str, ground_truth: str) -> float:
        """
        Compute structural similarity of proofs
        Rewards similar proof strategies (tactics, lemma usage)
        """
        # Extract tactics used
        sol_tactics = self._extract_tactics(solution)
        truth_tactics = self._extract_tactics(ground_truth)
        
        if not sol_tactics and not truth_tactics:
            return 1.0
        if not sol_tactics or not truth_tactics:
            return 0.0
        
        # Compute Jaccard similarity of tactics
        intersection = len(set(sol_tactics) & set(truth_tactics))
        union = len(set(sol_tactics) | set(truth_tactics))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Also consider tactic sequence similarity (edit distance)
        seq_similarity = self._sequence_similarity(sol_tactics, truth_tactics)
        
        return 0.5 * jaccard + 0.5 * seq_similarity
    
    def _compute_efficiency(self, solution: str, ground_truth: str) -> float:
        """
        Reward more efficient (shorter, simpler) proofs
        """
        sol_length = len(solution.split())
        truth_length = len(ground_truth.split())
        
        # Prefer solution that's not much longer than ground truth
        if sol_length <= truth_length:
            return 1.0
        
        ratio = truth_length / sol_length
        # Exponential decay for longer proofs
        return max(0.0, ratio ** 1.5)
    
    # Helper methods (simplified implementations)
    
    def _extract_type_signature(self, code: str) -> Optional[str]:
        """Extract the type signature from a theorem/def"""
        import re
        match = re.search(r'(?:theorem|def|lemma)\s+\w+\s*:\s*([^:=]+)', code)
        return match.group(1).strip() if match else None
    
    def _extract_theorem_statement(self, code: str) -> Optional[str]:
        """Extract the full theorem statement"""
        import re
        match = re.search(r'((?:theorem|def|lemma)\s+\w+[^:]*:[^:=]+)', code)
        return match.group(1).strip() if match else None
    
    def _extract_proof_term(self, code: str) -> Optional[str]:
        """Extract the proof term (right side of :=)"""
        import re
        match = re.search(r':=\s*(.+)', code, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_tactics(self, code: str) -> list:
        """Extract tactics used in proof"""
        import re
        tactics = re.findall(r'\b(intro|apply|exact|rfl|simp|rw|cases|induction|split|constructor|left|right|assumption|trivial|sorry)\b', code)
        return tactics
    
    def _sequence_similarity(self, seq1: list, seq2: list) -> float:
        """Compute normalized edit distance similarity"""
        # Levenshtein distance implementation
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len)
    
    def _are_alpha_equivalent(self, type1: str, type2: str) -> bool:
        """Check for α-equivalence (variable renaming)"""
        # Simplified - would need proper parsing
        # Remove whitespace and compare structure
        import re
        normalize = lambda s: re.sub(r'\s+', '', s)
        return normalize(type1) == normalize(type2)
    
    def _type_similarity(self, type1: str, type2: str) -> float:
        """Compute similarity between type signatures"""
        # Token-based similarity
        tokens1 = set(type1.split())
        tokens2 = set(type2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union


# Example usage
if __name__ == "__main__":
    reward_fn = Lean4RewardFunction()
    
    ground_truth = """
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n with
  | zero => simp
  | succ n ih => simp [Nat.add_succ, ih]
"""
    
    solution_str = """
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction m with
  | zero => simp [Nat.add_zero]
  | succ m ih => simp [Nat.succ_add, ih]
"""
    
    total_reward, components = reward_fn.evaluate(solution_str, ground_truth)
    
    print(f"Total Reward: {total_reward:.3f}")
    print(f"\nComponents:")
    print(f"  Syntactic Correctness: {components.syntactic_correctness:.3f}")
    print(f"  Type Correctness: {components.type_correctness:.3f}")
    print(f"  Semantic Equivalence: {components.semantic_equivalence:.3f}")
    print(f"  Structural Similarity: {components.structural_similarity:.3f}")
    print(f"  Efficiency: {components.efficiency:.3f}")


