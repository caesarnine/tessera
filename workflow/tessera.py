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
    
    def generate_mermaid(self) -> str:
        """
        Generate a Mermaid.js diagram representation of the workflow.
        Returns a string containing the Mermaid diagram definition.
        """
        # Track visited states to handle cycles
        visited_edges = set()
        
        # Collect all state metadata
        state_metadata = {}
        for state_name, state_func in self.states.items():
            # Get docstring if available
            doc = state_func.__doc__ or ""
            doc = doc.strip().split('\n')[0]  # Get first line of docstring
            
            state_metadata[state_name] = {
                'description': doc,
                'next_states': state_func.next_states
            }
        
        # Start building the diagram
        mermaid_lines = ['stateDiagram-v2']
        
        # Add states with descriptions
        for state_name, metadata in state_metadata.items():
            if metadata['description']:
                mermaid_lines.append(f'    {state_name}: {metadata["description"]}')
            else:
                mermaid_lines.append(f'    {state_name}')
        
        # Add transitions
        for state_name, metadata in state_metadata.items():
            for next_state in metadata['next_states']:
                edge = (state_name, next_state)
                if edge not in visited_edges:
                    mermaid_lines.append(f'    {state_name} --> {next_state}')
                    visited_edges.add(edge)
        
        return '\n'.join(mermaid_lines)
    
    def visualize(self, port: int = 0) -> None:
        """
        Generate and display the workflow diagram in a web browser.
        
        Args:
            port: Port to run the server on. If 0, uses a random available port.
        """
        # Generate the diagram
        diagram = self.generate_mermaid()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Workflow Visualization</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                #diagram {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div id="diagram">
                <pre class="mermaid">
                    {diagram}
                </pre>
            </div>
            <script>
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        </body>
        </html>
        """
        
        # Create a temporary file to serve
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / "workflow.html"
        temp_path.write_text(html_content)
        
        # Create custom handler that serves our file
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=temp_dir, **kwargs)
            
            def log_message(self, format, *args):
                # Suppress logging
                pass
        
        # Find an available port if none specified
        with socketserver.TCPServer(("", port), Handler) as httpd:
            server_port = httpd.server_address[1]
            
            # Start server in a separate thread
            server_thread = Thread(target=httpd.serve_forever)
            server_thread.daemon = True  # Thread will close when main program exits
            server_thread.start()
            
            # Open browser
            url = f"http://localhost:{server_port}/workflow.html"
            webbrowser.open(url)
            
            print(f"Visualization server running at {url}")
            print("Press Ctrl+C to stop the server...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down server...")
                httpd.shutdown()
                httpd.server_close()
    
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