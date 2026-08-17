from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.conversations.models import AgentContextSummary, AgentMessage, AgentSession, AgentSessionStatus


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self._session.get(AgentSession, session_id)

    def list_sessions(self, *, limit: int = 50, offset: int = 0, include_archived: bool = False) -> list[AgentSession]:
        statement = select(AgentSession)
        if not include_archived:
            statement = statement.where(AgentSession.status == AgentSessionStatus.ACTIVE)
        statement = statement.order_by(AgentSession.updated_at.desc(), AgentSession.created_at.desc()).offset(offset).limit(limit)
        return list(self._session.scalars(statement).all())

    def add_session(self, session: AgentSession) -> AgentSession:
        self._session.add(session)
        self._session.flush()
        return session

    def get_message(self, message_id: str) -> AgentMessage | None:
        return self._session.get(AgentMessage, message_id)

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        before_message_id: str | None = None,
    ) -> list[AgentMessage]:
        statement = select(AgentMessage).where(AgentMessage.session_id == session_id)
        if before_message_id is not None:
            before = self.get_message(before_message_id)
            if before is not None:
                statement = statement.where(AgentMessage.created_at < before.created_at)
        statement = statement.order_by(AgentMessage.created_at.asc(), AgentMessage.id.asc()).limit(limit)
        return list(self._session.scalars(statement).all())

    def list_uncompacted_messages(self, session_id: str) -> list[AgentMessage]:
        statement = (
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .where(AgentMessage.compacted_by_summary_id.is_(None))
            .where(AgentMessage.exclude_from_context.is_(False))
            .order_by(AgentMessage.created_at.asc(), AgentMessage.id.asc())
        )
        return list(self._session.scalars(statement).all())

    def add_message(self, message: AgentMessage) -> AgentMessage:
        self._session.add(message)
        self._session.flush()
        return message

    def get_summary(self, summary_id: str) -> AgentContextSummary | None:
        return self._session.get(AgentContextSummary, summary_id)

    def get_latest_summary(self, session_id: str) -> AgentContextSummary | None:
        return self._session.scalar(
            select(AgentContextSummary)
            .where(AgentContextSummary.session_id == session_id)
            .order_by(AgentContextSummary.created_at.desc(), AgentContextSummary.id.desc())
            .limit(1)
        )

    def add_summary(self, summary: AgentContextSummary) -> AgentContextSummary:
        self._session.add(summary)
        self._session.flush()
        return summary

    def flush(self) -> None:
        self._session.flush()
