"""InteractionLoop: drive one task to completion through Condition + UserSimulator.

Order of operations per task:
  1. simulator.initiate_task(task) -> initial user message
  2. while turn < max_turns:
       a. condition.handle_user_message(...) -> assistant response
       b. simulator.respond_to_assistant(...) -> (reply, violated_prefs, action)
       c. record both turns (mark assistant turn's violated_preferences,
          mark user turn's triggered_preferences)
       d. if action in {"accept", "give_up"}: break
  3. compute correction_count
  4. run test-case sanity (best-effort; never blocks)
  5. condition.finalize_task(task, interaction)
"""

from __future__ import annotations

from skillmap_eval.conditions.base import Condition
from skillmap_eval.metrics.correctness_sanity import (
    format_test_results_for_llm,
    run_sanity_check_for_task,
)
from skillmap_eval.simulator.user_simulator import UserSimulator
from skillmap_eval.types import (
    CompletionReason,
    ConditionName,
    CorrectionAxis,
    EvalTask,
    SimulatedTurn,
    TaskInteraction,
)


class InteractionLoop:
    def __init__(self, max_turns: int = 8, sanity_timeout_s: float = 10.0) -> None:
        self.max_turns = max_turns
        self.sanity_timeout_s = sanity_timeout_s

    async def run_single_task(
        self,
        task: EvalTask,
        task_index: int,
        condition: Condition,
        simulator: UserSimulator,
    ) -> TaskInteraction:
        turns: list[SimulatedTurn] = []

        # 1. Initial user message.
        initial_user_msg = await simulator.initiate_task(task)
        retrieved_ids_at_start: list[str] = []

        user_msg = initial_user_msg
        completion: CompletionReason = "max_turns_exceeded"
        violated_latest: list[str] = []
        axes_latest: list[CorrectionAxis] = []

        # We treat "one turn" as one (user -> assistant -> user-reaction) cycle.
        for turn_idx in range(self.max_turns):
            # Record the user message we are about to send.
            turns.append(
                SimulatedTurn(
                    role="user",
                    content=user_msg,
                    triggered_preferences=violated_latest if turn_idx > 0 else [],
                    correction_axes=axes_latest if turn_idx > 0 else [],
                )
            )
            # 2a. Assistant response.
            assistant_msg, retrieved = await condition.handle_user_message(
                task=task,
                conversation_so_far=turns[:-1],  # all prior, excluding the user_msg we just appended? actually the helper expects all-prior; include the just-appended user msg by passing turns[:-1] + fresh user_message argument. We pass turns[:-1] (prior) + user_msg as the new message.
                user_message=user_msg,
            )
            if turn_idx == 0:
                retrieved_ids_at_start = list(retrieved)

            # 2b. Run tests on this response (best-effort, never raises).
            test_results_str = ""
            try:
                test_results_str = format_test_results_for_llm(
                    task, assistant_msg, max_cases=2, timeout_s=5.0
                )
            except Exception:
                pass

            # 2c. Simulator reacts (with optional test results context).
            reply, violated, axes, action = await simulator.respond_to_assistant(
                task=task,
                conversation_so_far=turns,
                latest_assistant_message=assistant_msg,
                test_results=test_results_str,
            )

            # Record assistant turn with violated prefs attached.
            turns.append(
                SimulatedTurn(
                    role="assistant",
                    content=assistant_msg,
                    violated_preferences=violated,
                )
            )

            if action == "accept":
                if reply:
                    turns.append(
                        SimulatedTurn(
                            role="user",
                            content=reply,
                            triggered_preferences=[],
                            correction_axes=[],
                        )
                    )
                completion = "user_accepted"
                break
            if action == "give_up":
                if reply:
                    turns.append(
                        SimulatedTurn(
                            role="user",
                            content=reply,
                            triggered_preferences=violated,
                            # give_up is not itself a correction; leave axes empty
                            correction_axes=[],
                        )
                    )
                completion = "user_gave_up"
                break

            # action == "correct": set up the next iteration.
            user_msg = reply
            violated_latest = violated
            axes_latest = axes

        # Per-axis counters. A correction tagged with both axes increments
        # both counters, so the two can sum to MORE than correction_count.
        preference_correction_count = sum(
            1 for t in turns
            if t.role == "user" and "preference" in t.correction_axes
        )
        correctness_correction_count = sum(
            1 for t in turns
            if t.role == "user" and "correctness" in t.correction_axes
        )
        # Total = number of user turns that carried any correction axis.
        correction_count = sum(
            1 for t in turns if t.role == "user" and t.correction_axes
        )

        # 4. Sanity check (best-effort; never raises).
        pass_rate: float | None = None
        try:
            if turns:
                last_assistant = next(
                    (t.content for t in reversed(turns) if t.role == "assistant"),
                    None,
                )
                if last_assistant is not None:
                    pass_rate = run_sanity_check_for_task(
                        task, last_assistant, timeout_s=self.sanity_timeout_s
                    )
        except Exception:
            pass_rate = None

        interaction = TaskInteraction(
            task_id=task.task_id,
            condition_name=_cast_condition_name(condition.name),
            task_index_in_stream=task_index,
            turns=turns,
            correction_count=correction_count,
            preference_correction_count=preference_correction_count,
            correctness_correction_count=correctness_correction_count,
            completion_reason=completion,
            test_case_pass_rate=pass_rate,
            retrieved_skill_ids_at_start=retrieved_ids_at_start,
        )

        # 5. Let the condition update its memory.
        await condition.finalize_task(task=task, interaction=interaction)

        return interaction


def _cast_condition_name(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition name {name!r}")
    return name  # type: ignore[return-value]
