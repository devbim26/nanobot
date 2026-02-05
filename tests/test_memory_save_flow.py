from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse


@dataclass
class FakeProvider(LLMProvider):
    """Deterministic provider for tests."""

    draft: str = "# Резюме\n\n- тест"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = (messages, tools, model, max_tokens, temperature)
        return LLMResponse(content=self.draft)

    def get_default_model(self) -> str:
        return "fake"


def _make_agent(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = FakeProvider()
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path)


async def test_save_memory_draft_confirm_writes_file(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    session_key = "telegram:1"
    # Seed some dialog history.
    session = agent.sessions.get_or_create(session_key)
    session.add_message("user", "Привет")
    session.add_message("assistant", "Здравствуйте")
    agent.sessions.save(session)

    # Step 1: click button -> draft
    resp1 = await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content="Сохранить в памяти",
        )
    )
    assert resp1 is not None
    session = agent.sessions.get_or_create(session_key)
    assert session.metadata.get("awaiting_memory_confirm") is True
    assert "Резюме" in str(session.metadata.get("memory_draft"))

    # Step 2: confirm -> write to memory/MEMORY.md
    resp2 = await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content="сохранить",
        )
    )
    assert resp2 is not None

    memory_file = tmp_path / "memory" / "MEMORY.md"
    assert memory_file.exists()
    assert "Резюме" in memory_file.read_text(encoding="utf-8")

    session = agent.sessions.get_or_create(session_key)
    assert not session.metadata.get("awaiting_memory_confirm")
    assert "memory_draft" not in session.metadata


async def test_save_memory_cancel_clears_draft(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    session_key = "telegram:1"
    session = agent.sessions.get_or_create(session_key)
    session.add_message("user", "Привет")
    session.add_message("assistant", "Здравствуйте")
    agent.sessions.save(session)

    await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content="Сохранить в памяти",
        )
    )

    resp = await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content="Отмена",
        )
    )
    assert resp is not None

    session = agent.sessions.get_or_create(session_key)
    assert not session.metadata.get("awaiting_memory_confirm")
    assert "memory_draft" not in session.metadata


async def test_save_memory_editing_replaces_draft(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    session_key = "telegram:1"
    session = agent.sessions.get_or_create(session_key)
    session.add_message("user", "Привет")
    session.add_message("assistant", "Здравствуйте")
    agent.sessions.save(session)

    await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content="Сохранить в памяти",
        )
    )

    edited = "# Резюме\n\n- вручную"
    await agent._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u",
            chat_id="1",
            content=edited,
        )
    )

    session = agent.sessions.get_or_create(session_key)
    assert session.metadata.get("memory_draft") == edited

