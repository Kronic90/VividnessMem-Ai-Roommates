"""
test_task_recall.py — Tests for Aria's organic task recall system.

Verifies that task memories surface ONLY when the current conversation
naturally connects to past experiences, using human-style trigger
detection (not RAG / embeddings / similarity scores).

Tests:
  1. Simulation context triggers simulation task memories
  2. Coding context triggers code task memories
  3. Unrelated context returns nothing
  4. Low-importance auto-captured entries are filtered out
  5. Lesson-first output format
  6. Prefix matching handles plurals/conjugations
  7. Action type bonus works
  8. Touch-on-recall (accessing makes memory more vivid)
  9. Recall against LIVE data (real task_memories.json)
"""

import sys
import os
import unittest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_memory import AriaTaskMemory, TaskEntry


def _make_entry(summary, reflection="", keywords=None, importance=7,
                task_type="general", emotion=""):
    """Create a TaskEntry for testing."""
    return TaskEntry(
        summary=summary,
        reflection=reflection,
        keywords=keywords or [],
        task_type=task_type,
        importance=importance,
        timestamp="2025-07-01T12:00:00",
        emotion=emotion,
    )


class TestRecallFiltering(unittest.TestCase):
    """Recall should only surface memories connected to current context."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        self.mem.entries = [
            _make_entry(
                "Ran a population growth simulation with exponential parameters",
                "Growth rate was too high — need to cap at 5% for realistic results",
                ["simulation", "population", "growth", "parameters"],
                importance=8, task_type="general"
            ),
            _make_entry(
                "Wrote a Python function to calculate fibonacci sequence",
                "Recursive approach was too slow — memoization cut time by 90%",
                ["code", "python", "fibonacci", "algorithm"],
                importance=7, task_type="general"
            ),
            _make_entry(
                "Built world lore about the desert faction in Aetheria",
                "Desert nomads need more distinct cultural identity",
                ["worldbuild", "aetheria", "desert", "faction"],
                importance=9, task_type="general"
            ),
            _make_entry(
                "Saved session notes to file",
                "",
                ["file", "save"],
                importance=3, task_type="auto"  # auto-captured, low importance
            ),
        ]

    def test_simulation_context_triggers_simulation_memory(self):
        """Should recall the population simulation when discussing simulations."""
        context = "Let's run a simulation to model population growth in the settlements"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0, "Should recall simulation experiences")
        summaries = " ".join(e.summary for e in results)
        self.assertIn("population", summaries.lower())

    def test_coding_context_triggers_code_memory(self):
        """Should recall Python coding experience when discussing code."""
        context = "Can you write a Python function to compute something algorithmically?"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0, "Should recall coding experiences")
        summaries = " ".join(e.summary for e in results)
        self.assertIn("python", summaries.lower())

    def test_unrelated_context_returns_nothing(self):
        """Context about cooking should NOT trigger any task memories."""
        context = "What's a good recipe for chocolate cake? I love baking."
        results = self.mem.recall(context)
        self.assertEqual(len(results), 0, "Unrelated context should trigger no recall")

    def test_low_importance_filtered_out(self):
        """Auto-captured entries (imp < 5) should never surface."""
        context = "Let's save this document to a file for later"
        results = self.mem.recall(context)
        for entry in results:
            self.assertGreaterEqual(entry.importance, 5,
                                    f"Low-importance entry surfaced: {entry.summary}")

    def test_worldbuilding_context_triggers_lore_memory(self):
        """Should recall worldbuilding when discussing Aetheria lore."""
        context = "Tell me about the desert faction in Aetheria and their culture"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0, "Should recall worldbuilding experiences")
        summaries = " ".join(e.summary for e in results)
        self.assertIn("desert", summaries.lower())


class TestRecallFormat(unittest.TestCase):
    """recall_context_block() should produce lesson-first output."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        self.mem.entries = [
            _make_entry(
                "Simulated battle between two factions with varying strength parameters",
                "Overwhelming numbers matter less when terrain advantage is above 40%",
                ["simulation", "battle", "faction", "terrain"],
                importance=8
            ),
        ]

    def test_recall_block_leads_with_lesson(self):
        """Output should lead with 'I learned:' not just the summary."""
        context = "Let's simulate a battle between factions and see who wins"
        block = self.mem.recall_context_block(context)
        self.assertIn("I learned:", block)
        self.assertIn("terrain", block.lower())

    def test_recall_block_empty_for_unrelated(self):
        """No output when context doesn't connect."""
        block = self.mem.recall_context_block("Let's talk about music theory")
        self.assertEqual(block, "")

    def test_recall_block_includes_source(self):
        """Each lesson should reference what task it came from."""
        context = "We should run a simulation to test how faction battles play out"
        block = self.mem.recall_context_block(context)
        self.assertIn("from:", block.lower())


class TestPrefixMatching(unittest.TestCase):
    """Prefix matching should handle plurals and word forms."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        self.mem.entries = [
            _make_entry(
                "Analyzed simulation results for settlement viability",
                "Settlements below 50 population aren't self-sustaining",
                ["simulation", "settlement", "analysis", "viability"],
                importance=8
            ),
        ]

    def test_plural_form_matches(self):
        """'settlements' should match 'settlement' via prefix overlap."""
        context = "How viable are the settlements we've been analyzing?"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0, "Plural form should match via prefix")

    def test_conjugation_matches(self):
        """'simulating' + 'analyzing' should match 'simulation' + 'analysis' via prefix."""
        context = "I'm simulating and analyzing the settlement viability scenario"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0, "Conjugated forms should match via prefix")


class TestActionBonus(unittest.TestCase):
    """Action type matching should boost relevance when doing the same kind of task."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        # Two memories with similar keyword overlap but different action types
        self.mem.entries = [
            _make_entry(
                "Wrote code to simulate particle physics model",
                "The simulation diverged — needed smaller time steps",
                ["code", "simulate", "physics", "model"],
                importance=7
            ),
            _make_entry(
                "Analyzed the physics model results in detail",
                "Results were inconsistent because sample size was too small",
                ["analysis", "physics", "model", "results"],
                importance=7
            ),
        ]

    def test_code_context_prefers_code_memory(self):
        """When writing code about physics, code+simulation memory should rank higher."""
        context = "Write a Python script to simulate the physics model"
        results = self.mem.recall(context)
        self.assertTrue(len(results) > 0)
        # The code+simulation memory should come first due to action bonus
        self.assertIn("code", results[0].summary.lower())


class TestTouchOnRecall(unittest.TestCase):
    """Accessing a memory through recall should increase its access count."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        self.mem.entries = [
            _make_entry(
                "Created a simulation model for resource distribution",
                "Even distribution is suboptimal — weight by population density",
                ["simulation", "resource", "distribution"],
                importance=8
            ),
        ]

    def test_recall_increments_access_count(self):
        """Recalling a memory should touch it (increase access_count)."""
        entry = self.mem.entries[0]
        initial_count = entry._access_count
        context = "Let's create a simulation to model resource distribution"
        self.mem.recall(context)
        self.assertEqual(entry._access_count, initial_count + 1,
                         "Recall should increment access_count")


class TestLiveData(unittest.TestCase):
    """Test recall against Aria's real task memory data."""

    def setUp(self):
        self.mem = AriaTaskMemory()
        self.mem._load()

    def test_has_live_entries(self):
        """Should have real task memories on disk."""
        self.assertTrue(len(self.mem.entries) > 0,
                        "No live task memories found — is data present?")

    def test_recall_on_generic_query(self):
        """A broad engineering query should find something if there are deliberate entries."""
        # Check if we have any deliberate entries at all
        deliberate = [e for e in self.mem.entries if e.importance >= 5]
        if not deliberate:
            self.skipTest("No deliberate (imp>=5) entries in live data")

        # Try a query that uses keywords from a real entry
        sample = deliberate[0]
        context = f"Let's work on {sample.summary[:50]}"
        results = self.mem.recall(context)
        # Should find at least the entry we pulled keywords from
        self.assertTrue(len(results) > 0,
                        f"Failed to recall entry about: {sample.summary[:60]}")

    def test_garbage_query_returns_nothing(self):
        """Total gibberish should not match any real task memories."""
        context = "xyzzy plugh twisty passages maze different"
        results = self.mem.recall(context)
        self.assertEqual(len(results), 0,
                         f"Gibberish matched {len(results)} entries!")

    def test_no_low_importance_in_results(self):
        """Live recall should never surface auto-captured noise."""
        # Use keywords from an auto entry if available
        auto = [e for e in self.mem.entries if e.importance < 5]
        if not auto:
            self.skipTest("No auto entries in live data")
        context = auto[0].summary
        results = self.mem.recall(context)
        for r in results:
            self.assertGreaterEqual(r.importance, 5,
                                    f"Auto entry leaked: {r.summary[:60]}")


if __name__ == "__main__":
    print("=" * 70)
    print("  TASK RECALL TEST SUITE — Organic Trigger-Based Recall")
    print("  (No RAG, No Embeddings, No Similarity Scores)")
    print("=" * 70)
    print()
    unittest.main(verbosity=2)
