"""
VividnessMem — Organic Episodic Memory for AI Agents
=====================================================

    pip install vividnessmem
    pip install vividnessmem[encryption]   # for AES-at-rest

Quick start::

    from vividnessmem import VividnessMem

    mem = VividnessMem("./my_agent_memory")
    mem.add_self_reflection("I enjoyed that conversation", "happy", 8)
    matches = mem.resonate("that good conversation we had")
    context = mem.get_context_block()
    mem.save()
"""

from .vividnessmem import Memory, ShortTermFact, VividnessMem

__all__ = ["VividnessMem", "Memory", "ShortTermFact"]
__version__ = "1.0.4"
