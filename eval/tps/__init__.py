"""Temporal Policy Switching (TPS) benchmark.

Tests whether CI changes downstream agent policy when elapsed time
determines the correct action, with the visible prompt held fixed.

Primary endpoint: balanced policy accuracy under identical visible prompts
where only the hidden tau scalar differs.
"""
