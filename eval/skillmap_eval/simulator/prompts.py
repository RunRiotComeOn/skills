"""Prompts for the user simulator (LLM-A role). Placeholders use .format()."""

# Placeholders: {preference_profile_rendered}, {task_problem},
#               {conversation_rendered}, {latest_assistant_message},
#               {give_up_threshold}, {correction_style_hint}, {test_results_section}
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

{test_results_section}\
─────────────────────────────────────────────
STEP 1 — Identify issues (internal only)
Check TWO things:
a) Which preference IDs (if any) are violated by THIS response?
b) Did the code fail any tests (see test results above, if present)?

List both. Be precise; do not over-flag preferences.

STEP 1b — Tag the correction axes
For the response field below, you will set "correction_axes" to record what
KIND of correction this turn is. The valid axes are:
  "preference"  — your reply is correcting a style / format / approach
                  violation flagged in (a)
  "correctness" — your reply is correcting a code bug evidenced in (b)

If you only flag the issue from (a), axes = ["preference"].
If you only flag the issue from (b), axes = ["correctness"].
If your reply mentions BOTH a code bug AND a style violation, axes =
  ["preference", "correctness"].
If decision is "accept" or "give_up", set axes = [].

STEP 2 — Decide
  "correct"  — at least one preference violation OR the code failed tests
  "accept"   — no violations AND tests pass (or no tests ran) AND task looks done
  "give_up"  — same issue has been ignored {give_up_threshold}+ times and
               you've run out of patience (only when turn count > 5)

STEP 3 — Write your response
Rules for writing a natural, VARIED correction:

• If tests failed: mention it naturally as something YOU observed — "I tried
  running this and got X, but expected Y" or "this crashes on the first case".
  Do NOT mention "test cases" or "test results" as infrastructure — speak as
  a user who just ran the code themselves.

• Pick at most 1–2 of the most important issues to mention per turn.
  If both tests failed AND preferences were violated, lead with correctness.
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

• Keep your response under 60 words. Real user corrections are terse; do
  NOT write long explanations or traces. One or two sentences is enough.

{correction_style_hint}\
Output ONLY a JSON object (no prose, no code fence):
{{
  "violated_prefs": ["pref_id", ...],
  "correction_axes": ["preference" | "correctness", ...],
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
