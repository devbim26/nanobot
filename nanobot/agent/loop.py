"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from nanobot.agent.memory import MemoryStore
from nanobot.config.schema import ExecToolConfig

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.exec_config.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self, msg: InboundMessage
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # ------------------------------------------------------------------
        # Telegram reply-keyboard "buttons" are just text messages.
        # We implement a small state machine for saving the full dialog into
        # long-term memory (memory/MEMORY.md) with user confirmation.
        # ------------------------------------------------------------------
        normalized = (msg.content or "").strip().lower()
        is_save_memory = normalized in {"сохранить в памяти", "save to memory"}
        is_cancel_memory = normalized in {"отмена", "cancel"}
        is_confirm_save = normalized in {"сохранить", "save"}

        awaiting = bool(session.metadata.get("awaiting_memory_confirm"))
        if awaiting and is_cancel_memory:
            session.metadata.pop("awaiting_memory_confirm", None)
            session.metadata.pop("memory_draft", None)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Ок, отменил сохранение. Черновик сброшен.",
            )

        if awaiting and (is_confirm_save or is_save_memory):
            draft = str(session.metadata.get("memory_draft") or "").strip()
            if not draft:
                session.metadata.pop("awaiting_memory_confirm", None)
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        "Черновик не найден. Нажми «Сохранить в памяти» "
                        "ещё раз, чтобы сформировать новый черновик."
                    ),
                )

            MemoryStore(self.workspace).append_long_term(draft)
            session.metadata.pop("awaiting_memory_confirm", None)
            session.metadata.pop("memory_draft", None)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Сохранено в долгосрочную память (memory/MEMORY.md).",
            )

        if awaiting and not (
            is_confirm_save or is_save_memory or is_cancel_memory
        ):
            # Treat any text as an edited draft.
            edited = (msg.content or "").strip()
            if edited:
                session.metadata["memory_draft"] = edited
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        "Обновил черновик.\n\n"
                        f"{edited}\n\n"
                        "Теперь напиши «сохранить», нажми\n"
                        "«Сохранить в памяти» ещё раз, или нажми «Отмена»."
                    ),
                )

        if is_save_memory and not awaiting:
            # Build a transcript from the whole session (user + assistant).
            transcript_lines: list[str] = []
            for m in session.messages:
                role = m.get("role", "")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    transcript_lines.append(f"Пользователь: {content}")
                elif role == "assistant":
                    transcript_lines.append(f"Ассистент: {content}")
                else:
                    transcript_lines.append(f"{role}: {content}")

            transcript = "\n".join(transcript_lines)
            if not transcript.strip():
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Пока нечего сохранять: история диалога пуста.",
                )

            # Ask the LLM to draft a structured memory note.
            memory_prompt = (
                "Сформируй краткое структурированное резюме всего диалога для "
                "долгосрочной памяти. Требования:\n"
                "- Пиши по-русски\n"
                "- Только то, что действительно важно помнить\n"
                "- Структура Markdown: Заголовок, затем блоки: 'Факты', "
                "'Предпочтения', 'Контекст проекта', "
                "'Решения/договорённости', "
                "'Следующие шаги'\n"
                "- Если раздел пустой — пропусти его\n\n"
                "Диалог:\n"
                f"{transcript}"
            )

            draft_resp = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — ассистент, который готовит заметки в память."
                        ),
                    },
                    {"role": "user", "content": memory_prompt},
                ],
                tools=None,
                model=self.model,
            )

            draft = (draft_resp.content or "").strip()
            if not draft:
                draft = "(Не удалось сгенерировать черновик резюме.)"

            session.metadata["awaiting_memory_confirm"] = True
            session.metadata["memory_draft"] = draft
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "Черновик резюме для памяти:\n\n"
                    f"{draft}\n\n"
                    "Отправь правки текстом, или напиши «сохранить», "
                    "или нажми «Сохранить в памяти» ещё раз. "
                    "Чтобы отменить — нажми «Отмена»."
                ),
            )
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # Must be JSON string
                            "arguments": json.dumps(tc.arguments),
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                
                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(
                        "Executing tool: "
                        f"{tool_call.name} with arguments: {args_str}"
                    )
                    result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = (
                "I've completed processing but have no response to give."
            )
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
        )
    
    async def _process_system_message(
        self, msg: InboundMessage
    ) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(
                        "Executing tool: "
                        f"{tool_call.name} with arguments: {args_str}"
                    )
                    result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self, content: str, session_key: str = "cli:direct"
    ) -> str:
        """
        Process a message directly (for CLI usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
