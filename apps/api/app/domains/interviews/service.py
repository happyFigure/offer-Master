from __future__ import annotations


class InterviewService:
    deferred_until = "interview_phase"

    def queue_practice(self, *_args, **_kwargs):
        raise NotImplementedError("Interview practice is deferred until the interview phase")
