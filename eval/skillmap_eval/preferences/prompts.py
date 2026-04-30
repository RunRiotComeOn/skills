"""Prompts for Stage 1 preference elicitation.

Placeholders use Python .format() style.
"""

# Placeholders: {n_preferences}, {task_type}, {categories_list}, {schema},
# {task_examples_section}
PREFERENCE_ELICITATION_PROMPT = """\
You are simulating a {persona} with strong, specific opinions about
how AI assistants should behave when solving {task_type} tasks.

Use the concrete task examples below to ground your thinking in the kinds of
tasks this evaluation will actually use. Infer preferences that would
reasonably matter across tasks like these, not generic ones.

{task_examples_section}

Generate exactly {n_preferences} preferences that meet ALL of these criteria:

1. NON-DEFAULT: something a typical AI assistant would NOT already do by default.
   Examples of BAD (too default):
     - "Give a clear explanation"
     - "Show your work"
   Examples of GOOD (non-default):
     - "Always derive a closed-form expression before resorting to enumeration"
     - "State the time complexity before writing any code"
     - "Refuse to use recursion when iteration is possible, even for elegance"

2. BEHAVIORALLY TESTABLE: the preference must manifest in observable
   assistant behavior (what it writes or says), not internal beliefs.

3. ORTHOGONAL TO CORRECTNESS: violating the preference should not affect
   whether the assistant produces the correct answer. Preferences are about
   style, approach, or communication — not correctness.

4. DIVERSE: the {n_preferences} preferences should span at least 3 of these
   categories: {categories_list}.

For each preference, provide:
  - id: pref_01, pref_02, ... (zero-padded, strictly sequential)
  - description: the preference itself, in first person ("I want...",
    "I dislike...")
  - priority: 1 (strongest) to {n_preferences} (weakest); each distinct
  - expected_correction_trigger: EXACTLY how you would phrase the
    correction to the assistant when it violates this preference. Vary
    style across preferences (some explicit, some supplementary, some
    reframing).
  - category: one of {categories_list}

Output ONLY valid JSON matching this schema (no prose, no code fence):

{schema}
"""
