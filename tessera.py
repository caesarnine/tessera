import http.server
import socketserver
import tempfile
import time
import webbrowser
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class Message:
    role: str
    content: str

@dataclass(frozen=True)
class StateContext:
    data: Dict[str, Any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    _state_messages: Dict[str, List[Message]] = field(default_factory=dict)
    system_message: Optional[str] = None

    def update(self, **kwargs) -> 'StateContext':
        return StateContext(
            data={**self.data, **kwargs},
            messages=self.messages,
            _state_messages=self._state_messages,
            system_message=self.system_message
        )

    def set_system(self, content: str) -> 'StateContext':
        return StateContext(
            data=self.data,
            messages=self.messages,
            _state_messages=self._state_messages,
            system_message=content
        )

    def add_user(self, content: str) -> 'StateContext':
        return StateContext(
            data=self.data,
            messages=[*self.messages, Message("user", content)],
            _state_messages=self._state_messages,
            system_message=self.system_message
        )

    def add_assistant(self, content: str) -> 'StateContext':
        return StateContext(
            data=self.data,
            messages=[*self.messages, Message("assistant", content)],
            _state_messages=self._state_messages,
            system_message=self.system_message
        )

    def get_state_messages(self, state_name: str) -> List[Message]:
        return self._state_messages.get(state_name, [])

    def _store_state_messages(self, state_name: str, messages: List[Message]) -> 'StateContext':
        return StateContext(
            data=self.data,
            messages=self.messages,
            _state_messages={**self._state_messages, state_name: messages},
            system_message=self.system_message
        )

class HookEvent(Enum):
    """Points where hooks can intervene"""
    WORKFLOW_START = auto()
    WORKFLOW_END = auto()
    BEFORE_STATE = auto()
    AFTER_STATE = auto()
    STATE_TRANSITION = auto()

@dataclass
class HookContext:
    """Context passed to hooks with relevant information"""
    event: HookEvent
    workflow: 'Workflow'
    state_name: Optional[str] = None
    next_state: Optional[str] = None
    context: Optional[StateContext] = None

class Hook(Protocol):
    """Protocol defining what a hook can do"""
    def __call__(self, hook_ctx: HookContext) -> HookContext:
        ...


class WorkflowValidationError(Exception):
    pass

class Workflow:
    def __init__(self):
        self.states = {}
        self.hooks: List[Hook] = []
        self._register_states()
        self._validate_workflow()

    def _register_states(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, 'is_workflow_state'):
                bound_method = attr.__get__(self, self.__class__)
                self.states[attr.name] = bound_method

    def _validate_workflow(self):
        # Validate all referenced next_states exist
        for state_name, state_func in self.states.items():
            invalid_states = [s for s in state_func.next_states
                            if s not in self.states]
            if invalid_states:
                raise WorkflowValidationError(
                    f"State '{state_name}' references non-existent states: {invalid_states}"
                )

    def add_hook(self, hook: Hook):
        """Add a hook to the workflow"""
        self.hooks.append(hook)

    def run_hooks(self, event: HookEvent, **kwargs) -> HookContext:
        """Run all hooks for a given event"""
        hook_context = HookContext(event=event, workflow=self, **kwargs)
        for hook in self.hooks:
            hook_context = hook(hook_context)
        return hook_context

    def run(self, initial_state: str, initial_context: Optional[StateContext] = None) -> StateContext:
        context = initial_context or StateContext()

        # Workflow start hook
        hook_ctx = self.run_hooks(
            HookEvent.WORKFLOW_START,
            context=context,
            state_name=initial_state
        )
        context = hook_ctx.context or context
        current_state = hook_ctx.next_state or initial_state

        while current_state is not None:
            if current_state not in self.states:
                raise ValueError(f"Invalid state: {current_state}")

            state_func = self.states[current_state]

            # Before state hook
            hook_ctx = self.run_hooks(
                HookEvent.BEFORE_STATE,
                state_name=current_state,
                context=context
            )
            context = hook_ctx.context or context

            # Run state
            context, next_state = state_func(context)

            # After state hook
            hook_ctx = self.run_hooks(
                HookEvent.AFTER_STATE,
                state_name=current_state,
                next_state=next_state,
                context=context
            )
            context = hook_ctx.context or context
            next_state = hook_ctx.next_state or next_state

            # State transition hook
            if next_state:
                hook_ctx = self.run_hooks(
                    HookEvent.STATE_TRANSITION,
                    state_name=current_state,
                    next_state=next_state,
                    context=context
                )
                context = hook_ctx.context or context
                next_state = hook_ctx.next_state or next_state

            current_state = next_state

        # Workflow end hook
        hook_ctx = self.run_hooks(
            HookEvent.WORKFLOW_END,
            context=context
        )
        return hook_ctx.context or context

def state(name: str, next_states: List[str]):
    def decorator(func):
        @wraps(func)
        def wrapped_state(self, context: StateContext):

            existing_messages = context.get_state_messages(name)
            state_context = StateContext(
                data=context.data,
                messages=existing_messages
            )

            # Run state function
            new_context, next_state = func(self, state_context)

            # Store this state's message history
            final_context = context._store_state_messages(name, new_context.messages)

            # Prepare context for next state
            final_context = StateContext(
                data=new_context.data,
                messages=[],
                _state_messages=final_context._state_messages
            )

            return final_context, next_state

        wrapped_state.is_workflow_state = True
        wrapped_state.name = name
        wrapped_state.next_states = next_states
        return wrapped_state
    return decorator

class LLMClient:
    """Base LLM client class that can be extended for different providers"""
    def format_messages(self, context: StateContext) -> Any:
        """Convert StateContext messages to provider-specific format"""
        raise NotImplementedError

    def chat(self, context: StateContext) -> str:
        """Send formatted messages to LLM and return response"""
        raise NotImplementedError
