from __future__ import annotations


class InterviewRepository:
    deferred_until = "interview_phase"

    def get_session(self, session_id: str):
        raise NotImplementedError(
            f"Interview persistence is deferred until the interview phase: {session_id}"
        )
