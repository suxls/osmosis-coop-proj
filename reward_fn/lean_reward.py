import re
from osmosis_ai import osmosis_reward


@osmosis_reward
def lean_reward(solution_str: str, ground_truth: str, extra_info: dict=None, **kwargs):
    return 1.0