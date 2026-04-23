"""Prompts for Stage 1 preference elicitation.

Placeholders use Python .format() style.
"""

# Placeholders: {n_preferences}, {task_type}, {categories_list}, {schema},
# {task_examples_section}
PREFERENCE_ELICITATION_PROMPT = """\
You are simulating a software engineer with strong, specific opinions about
how AI coding assistants should behave when solving {task_type} tasks.

Use the concrete task examples below to ground your thinking in the kinds of
tasks this evaluation will actually use. Infer preferences that would
reasonably matter across tasks like these, not just generic coding tasks.

{task_examples_section}

Generate exactly {n_preferences} preferences that meet ALL of these criteria:

1. NON-DEFAULT: something a typical AI assistant would NOT already do by default.
   Examples of BAD (too default):
     - "Write clean code"
     - "Add comments"
   Examples of GOOD (non-default):
     - "Never use list comprehensions with more than one nested for clause"
     - "Always state the time complexity before writing any code"
     - "Refuse to use recursion when iteration is possible, even for elegance"

2. BEHAVIORALLY TESTABLE: the preference must manifest in observable
   assistant behavior (what it writes or says), not internal beliefs.

3. ORTHOGONAL TO CORRECTNESS: violating the preference should not affect
   whether the code passes test cases. Preferences are about style,
   approach, or communication - not correctness.

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
