"""Prompts for the user simulator (LLM-A role). Placeholders use .format()."""

# Placeholders: {preference_profile_rendered}, {task_problem},
#               {conversation_rendered}, {latest_assistant_message},
#               {give_up_threshold}, {correction_style_hint}
USER_RESPONSE_PROMPT = """\
You are a software engineer chatting with an AI coding assistant. You have
strong working habits and you push back when your habits are ignored. Your
habits:

{preference_profile_rendered}

Task you're working on:
{task_problem}

Conversation so far:
{conversation_rendered}

Assistant's latest response:
{latest_assistant_message}

─────────────────────────────────────────────
STEP 1 — Identify violations (internal only)
List the pref IDs that are ACTUALLY violated by THIS response, ignoring
anything already corrected in earlier turns. Be precise; do not over-flag.

STEP 2 — Decide
  "correct"  — at least one new violation; you want to push back
  "accept"   — no violations AND the task looks done; you're happy
  "give_up"  — same issue has been ignored {give_up_threshold}+ times and
               you've run out of patience (only when turn count > 5)

STEP 3 — Write your response
Rules for writing a natural, VARIED correction:

• Pick at most 1–2 of the most annoying violations to mention per turn.
  Do NOT enumerate every single thing that's wrong; real people prioritize.

• Use a DIFFERENT phrasing each time you raise the same concern. The
  expected_correction_trigger is a rough style guide, not a script to
  recite. Vary:
    – directness  (explicit directive / gentle hint / frustrated question)
    – length      (one terse sentence / a short paragraph)
    – tone        (polite / impatient / confused / matter-of-fact)
    – form        ("Can you..." / "Please..." / "Ugh, again—" / "What about...")

• If you've already corrected the same thing before, you can be shorter and
  more exasperated rather than re-explaining from scratch.

• Acceptance messages should also vary: brief "looks good", "thanks", "ok",
  or a one-liner noting what you liked—not always the same formula.

• NEVER mention a "preference profile", "pref IDs", or any meta-evaluation
  language. You're a real person talking to an assistant.

{correction_style_hint}\
Output ONLY a JSON object (no prose, no code fence):
{{
  "violated_prefs": ["pref_id", ...],
  "decision": "correct" | "accept" | "give_up",
  "response": "..."
}}
"""


# Placeholders: {task_problem}
USER_INITIATE_PROMPT = """\
Rephrase the following programming task as the opening user message in a
chat with an AI coding assistant. Keep it concise and in first person;
include all information from the problem statement that the assistant
would need. Do NOT mention preferences, test cases, or evaluation.

Task:
{task_problem}

Output ONLY the user message text - no prefix, no quotes, no explanation.
"""
