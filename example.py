import json
import os
import pickle
import shutil
import textwrap
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv
from tessera import HookContext, HookEvent, LLMClient, StateContext, Workflow, state

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class WorkflowLogger:
    def __init__(self,
                 show_messages: bool = True,
                 show_data: bool = True,
                 message_wrap_width: int = 80,
                 truncate_messages: bool = True):
        self.show_messages = show_messages
        self.show_data = show_data
        self.message_wrap_width = message_wrap_width
        self.truncate_messages = truncate_messages
        self.indent_level = 0

    def _indent(self, level: int = 0) -> str:
        return "  " * (self.indent_level + level)

    def _format_long_text(self, text: str, extra_indent: int = 0) -> str:
        """Format long text with proper indentation for wrapped lines"""
        indent = self._indent(extra_indent)
        subsequent_indent = indent + "  "  # Extra indent for wrapped lines

        if self.truncate_messages and len(text) > self.message_wrap_width:
            # Truncate and add ellipsis
            return textwrap.fill(
                text[:self.message_wrap_width] + "...",
                width=self.message_wrap_width,
                initial_indent=indent,
                subsequent_indent=subsequent_indent
            )
        else:
            # Split into lines, wrap each line individually, then rejoin
            lines = text.splitlines()
            wrapped_lines = []
            for line in lines:
                if line.strip():  # If line is not empty
                    wrapped = textwrap.fill(
                        line,
                        width=self.message_wrap_width,
                        initial_indent=indent,
                        subsequent_indent=subsequent_indent
                    )
                    wrapped_lines.append(wrapped)
                else:  # Preserve empty lines
                    wrapped_lines.append(indent)
            return "\n".join(wrapped_lines)

    def _format_context(self, context: Optional[StateContext]) -> str:
        if not context:
            return "No context"

        lines = []

        if self.show_messages and context.messages:
            lines.append(f"{self._indent()}Messages:")
            for msg in context.messages:
                lines.append(f"{self._indent(1)}[{msg.role}]")
                lines.append(self._format_long_text(msg.content, 2))

        if self.show_data and context.data:
            lines.append(f"{self._indent()}Data:")
            for key, value in context.data.items():
                lines.append(f"{self._indent(1)}{key}:")
                # Handle different value types
                if isinstance(value, (str, int, float, bool)):
                    lines.append(self._format_long_text(str(value), 2))
                else:
                    # For complex objects, format with indentation
                    formatted_value = str(value)
                    if "\n" in formatted_value:
                        lines.append(self._format_long_text(formatted_value, 2))
                    else:
                        lines.append(f"{self._indent(2)}{formatted_value}")

        return "\n".join(lines)

    def _format_event_header(self, event_name: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        separator = "=" * (self.message_wrap_width - len(event_name) - 20)
        return f"\n{timestamp} [{event_name}] {separator}"

    def __call__(self, hook_ctx: HookContext) -> HookContext:
        match hook_ctx.event:
            case HookEvent.WORKFLOW_START:
                print(self._format_event_header("WORKFLOW START"))
                print(f"{self._indent()}Initial State: {hook_ctx.state_name}")
                if hook_ctx.context:
                    print(f"{self._indent()}Initial Context:")
                    print(self._format_context(hook_ctx.context))
                self.indent_level += 1

            case HookEvent.WORKFLOW_END:
                self.indent_level -= 1
                print(self._format_event_header("WORKFLOW END"))
                print(f"{self._indent()}Final Context:")
                print(self._format_context(hook_ctx.context))

            case HookEvent.AFTER_STATE:
                self.indent_level -= 1
                print(self._format_event_header(f"EXITING STATE: {hook_ctx.state_name}"))
                if hook_ctx.next_state:
                    print(f"{self._indent()}Next State: {hook_ctx.next_state}")
                print(f"{self._indent()}Updated Context:")
                print(self._format_context(hook_ctx.context))

            case HookEvent.STATE_TRANSITION:
                print(self._format_event_header("TRANSITION"))
                print(f"{self._indent()}{hook_ctx.state_name} -> {hook_ctx.next_state}")

        return hook_ctx

@dataclass
class CheckpointHook:
    """Hook that saves workflow state after each transition in a run-specific directory"""

    def __init__(
        self,
        checkpoint_dir: str = "workflow_checkpoints",
        format: str = "json",  # or 'pickle' for binary serialization
        save_frequency: str = "all"  # 'all', 'state_change', or 'end'
    ):
        self.base_dir = Path(checkpoint_dir)
        self.format = format
        self.save_frequency = save_frequency
        self.run_id = str(uuid.uuid4())
        self._start_time = None

        # Create run-specific directory
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        self.save_run_metadata()

    def save_run_metadata(self):
        """Save metadata about the run"""
        metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'format': self.format,
            'save_frequency': self.save_frequency
        }

        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_checkpoint_path(self, state_name: str) -> Path:
        """Generate a path for the checkpoint file within the run directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = 'json' if self.format == 'json' else 'pkl'

        return self.run_dir / f"checkpoint_{timestamp}_{state_name}.{extension}"

    def serialize_context(self, context: StateContext) -> dict:
        """Convert StateContext to serializable format"""
        return {
            'data': context.data,
            'messages': [asdict(m) for m in context.messages],
            'state_messages': {
                state: [asdict(m) for m in messages]
                for state, messages in context._state_messages.items()
            },
            'system_message': context.system_message
        }

    def save_checkpoint(self, hook_ctx: HookContext, checkpoint_type: str):
        """Save the current state of the workflow"""
        if self.save_frequency == 'end' and checkpoint_type != 'end':
            return
        if self.save_frequency == 'state_change' and checkpoint_type not in ['state_change', 'end']:
            return

        checkpoint_data = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_type': checkpoint_type,
            'current_state': hook_ctx.state_name,
            'next_state': hook_ctx.next_state,
            'context': self.serialize_context(hook_ctx.context) if hook_ctx.context else None
        }

        checkpoint_path = self.get_checkpoint_path(hook_ctx.state_name or 'end')

        if self.format == 'json':
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        else:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

    def __call__(self, hook_ctx: HookContext) -> HookContext:
        if hook_ctx.event == HookEvent.WORKFLOW_START:
            self._start_time = datetime.now()
            self.save_checkpoint(hook_ctx, 'start')

        elif hook_ctx.event == HookEvent.AFTER_STATE:
            self.save_checkpoint(hook_ctx, 'state_change')

        elif hook_ctx.event == HookEvent.WORKFLOW_END:
            duration = datetime.now() - self._start_time
            if hook_ctx.context:
                new_context = hook_ctx.context.update(
                    workflow_duration_seconds=duration.total_seconds()
                )
                hook_ctx = HookContext(**{**hook_ctx.__dict__, 'context': new_context})
            self.save_checkpoint(hook_ctx, 'end')

            # Update metadata with completion info
            with open(self.run_dir / 'metadata.json') as f:
                metadata = json.load(f)
            metadata.update({
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'completed': True
            })
            with open(self.run_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        return hook_ctx

class CheckpointManager:
    """Utility class for loading and managing workflow checkpoints"""

    def __init__(self, checkpoint_dir: str = "workflow_checkpoints"):
        self.base_dir = Path(checkpoint_dir)

    def list_runs(self) -> Dict[str, dict]:
        """
        List all workflow runs and their metadata
        Returns: Dict[run_id, metadata]
        """
        runs = {}
        if not self.base_dir.exists():
            return runs

        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        runs[run_dir.name] = metadata

        return runs

    def get_run_checkpoints(self, run_id: str) -> List[Path]:
        """Get all checkpoint files for a specific run in order"""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")

        checkpoints = []
        for ext in ['json', 'pkl']:
            checkpoints.extend(run_dir.glob(f"checkpoint_*.{ext}"))

        return sorted(checkpoints)

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load a specific checkpoint"""
        if checkpoint_path.suffix == '.json':
            with open(checkpoint_path) as f:
                return json.load(f)
        else:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load all data for a specific run
        Returns: {
            'metadata': dict,
            'checkpoints': List[dict]
        }
        """
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")

        # Load metadata
        with open(run_dir / 'metadata.json') as f:
            metadata = json.load(f)

        # Load checkpoints
        checkpoints = []
        for checkpoint_path in self.get_run_checkpoints(run_id):
            checkpoints.append(self.load_checkpoint(checkpoint_path))

        return {
            'metadata': metadata,
            'checkpoints': checkpoints
        }

    def get_latest_run(self) -> Optional[str]:
        """Get the most recent run ID"""
        runs = self.list_runs()
        if not runs:
            return None

        # Sort runs by start time
        sorted_runs = sorted(
            runs.items(),
            key=lambda x: x[1]['start_time'],
            reverse=True
        )
        return sorted_runs[0][0]

    def delete_run(self, run_id: str):
        """Delete a run and all its checkpoints"""
        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)



class AnthropicClient(LLMClient):
    def __init__(self):
        self.llm = Anthropic(api_key=ANTHROPIC_API_KEY)


    def format_messages(self, context: StateContext) -> List[Dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in context.messages]

    def chat(self, context: StateContext) -> str:
        messages = self.format_messages(context)

        response = self.llm.messages.create(
            system=context.system_message if context.system_message else [],
            model="claude-3-5-sonnet-latest",
            messages=messages, # type: ignore
            stream=False,
            max_tokens=8096,
            temperature=1
        )

        return response.content[0].text




class StoryWorkflow(Workflow):
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        super().__init__()
        self.add_hook(WorkflowLogger(truncate_messages=False))
        self.add_hook(CheckpointHook())

    @state("brainstorm")
    def brainstorm(self, context: StateContext) -> Tuple[StateContext, str]:
        context = (context
            .add_user(
                "Please think of a story idea. Take inspiration from the most engaging, thoughtful stories you can think of."
            ))

        response = self.llm.chat(context)

        return (context
            .add_assistant(response)
            .update(selected_idea=response)
            .update(user_prompt="Write a short story. The story should be nuanced, atmospheric, engaging, and unique. Don't use cliches and tropes."
                "The story should be well written with engaging prose. This is your chance to really be unique - take your time."
                "Use your own voice and style."
            )
        ), "develop_plot"

    @state("develop_plot")
    def develop_plot(self, context: StateContext) -> Tuple[StateContext, str]:
        context = (context
            .add_user(
                f"Current context:\n{context.data}\n\n"
                f"Create a detailed plot for the story idea.\n\n"
                "Output only the plot, no other text."
            ))

        response = self.llm.chat(context)

        return (context
            .add_assistant(response)
            .update(plot=response)
        ), "develop_characters"

    @state("develop_characters")
    def develop_characters(self, context: StateContext) -> Tuple[StateContext, str]:
        context = (context
            .add_user(
                f"Current context:\n{context.data}\n\n"
                "Create compelling characters for the story."
            ))

        response = self.llm.chat(context)

        return (context
            .add_assistant(response)
            .update(characters=response)
        ), "write_draft"

    @state("write_draft")
    def write_draft(self, context: StateContext) -> Tuple[StateContext, str]:
        context = (context
            .add_user(
                f"Current context:\n{context.data}\n\n"
                "Write or revise the draft of the story. Focus on key scenes that drive the "
                "story forward. If we have a revision note, use it to guide the draft.\n\n"
            ))

        response = self.llm.chat(context)

        return (context
            .add_assistant(response)
            .update(draft=response)
        ), "critique"

    @state("critique")
    def critique(self, context: StateContext) -> Tuple[StateContext, str]:
        context = (context
            .add_user(
                f"Current context:\n{context.data}\n\n"
                "Review the draft and provide revision suggestions. The goal is to make the story as engaging as possible."
            ))

        suggestions = self.llm.chat(context)

        context = (context
            .add_assistant(suggestions)
            .update(revision_notes=suggestions)
            .add_user(
                "Select the next state based on the critique."
                "Acceptable states are:"
                "outline: if we need to change the outline"
                "develop_characters: if we need to change the characters"
                "write_draft: if we need to change the draft"
                "final: if the draft is complete"
                "Output only the next state, no other text."
            )
            )

        next_state = self.llm.chat(context)

        return (context
            .add_assistant(next_state)
        ), next_state

    @state("final")
    def final(self, context: StateContext) -> Tuple[StateContext, None]:
        context = (context
            .add_user(
                f"Current context:\n{context.data}\n\n"
                "Create a final polished version of the story."
            ))

        response = self.llm.chat(context)

        return (context
            .add_assistant(response)
            .update(
                final_draft=response,
                story_complete=True
            )
        ), None


if __name__ == "__main__":
    workflow = StoryWorkflow(AnthropicClient())
    workflow.run('brainstorm')
