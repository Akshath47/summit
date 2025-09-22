# Summit Architecture Documentation

## Overview

Summit is a multi-agent conversational AI system built using LangGraph that manages tasks and schedules through natural language interaction. The system uses specialized agents to handle different aspects of task management while maintaining persistent memory and providing personalized responses.

## System Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Conversation Agent │───▶│  Agent Routing  │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               │                               │
           ┌────────────▼────────────┐    ┌─────────────▼─────────────┐    ┌────────────▼────────────┐
           │   Task Manager Agent   │    │   Scheduler Agent         │    │   Profile Update Agent │
           │                        │    │                           │    │                        │
           │ - Add/Update Tasks     │    │ - Schedule Events         │    │ - Learn User Profile   │
           │ - Mark Complete        │    │ - Detect Conflicts        │    │ - Store Preferences    │
           │ - Generate Micro-steps │    │ - Color-code Priority     │    │                        │
           └────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
                        │                               │                               │
           ┌────────────▼────────────┐    ┌─────────────▼─────────────┐    ┌────────────▼────────────┐
           │   Focus Coach Agent    │    │ Instructions Update Agent │    │ Response Synthesizer  │
           │                        │    │                           │    │                        │
           │ - Analyze Mood/Energy  │    │ - Learn Task Preferences  │    │ - Generate Natural     │
           │ - Suggest Next Actions │    │ - Update Rules            │    │   Response            │
           │ - Provide Motivation   │    └─────────────────────────┘    └─────────────────────────┘
           └────────────────────────┘
```

## Agent Details

### 1. Conversation Agent (Router)

**Purpose**: Central dispatcher that parses user messages and routes to appropriate specialized agents.

**Key Components**:
- **Router LLM**: Uses structured output to decide disposition (dispatch, reply, clarify) and targets
- **Memory Integration**: Retrieves user profile, todos, events, and custom instructions
- **Smart Context**: Includes relevant conversation history for clarification/reply scenarios

**Decision Logic**:
- **Dispatch**: Routes to 1+ specialized agents for task management
- **Reply**: Direct response for questions about existing data
- **Clarify**: Ask for missing information before acting

**Output**: RouterDecision with targets and conversational acknowledgment

### 2. Task Manager Agent

**Purpose**: Manages ToDo items with persistent storage and intelligent micro-step generation.

**Key Features**:
- **Trustcall Integration**: Uses structured extraction for task CRUD operations
- **Solutions Generation**: Automatically creates 3-6 concrete micro-steps for new tasks
- **Memory Persistence**: Stores tasks in LangGraph Store with user namespaces

**Task Schema**:
```python
class ToDo(BaseModel):
    id: str
    task: str
    time_to_complete: Optional[str]
    deadline: Optional[datetime]
    priority: Literal["low", "medium", "high"]
    difficulty: Literal["easy", "medium", "hard"]
    solutions: List[str]  # Micro-steps
    status: Literal["not started", "in progress", "done", "archived"]
```

**Process Flow**:
1. Retrieve existing tasks from store
2. Extract updates via Trustcall
3. Generate solutions if missing
4. Save to persistent memory
5. Return summary for synthesizer

### 3. Scheduler Agent

**Purpose**: Handles calendar events and deadline management.

**Key Features**:
- **Event Creation**: Links tasks to scheduled times
- **Conflict Detection**: Identifies overlapping events (TODO: implement)
- **Color Coding**: Visual priority/difficulty indicators

**Event Schema**:
```python
class Event(BaseModel):
    id: str
    color: str  # UI color code
    title: str
    time: datetime
    task_id: Optional[str]  # Link to ToDo
    location: Optional[str]
    notes: Optional[str]
```

### 4. Profile Update Agent

**Purpose**: Learns and maintains user profile information.

**Profile Schema**:
```python
class Profile(BaseModel):
    name: Optional[str]
    location: Optional[str]
    job: Optional[str]
    college: Optional[str]
    course: Optional[str]
    interests: List[str]
```

### 5. Focus Coach Agent

**Purpose**: Provides personalized suggestions based on mood and energy levels.

**Process**:
1. Analyzes conversation for mood/energy signals
2. Reviews current task catalog
3. Selects appropriate next action:
   - Low energy: Smallest micro-step
   - Medium energy: Moderate priority task
   - High energy: High-impact or deadline-driven task
4. Returns FocusSuggestion with action and motivation

### 6. Instructions Update Agent

**Purpose**: Learns user preferences for how tasks should be managed.

**Example Preferences**:
- "Always ask before adding new tasks"
- "Mark urgent tasks as high priority"
- "Add estimated time automatically"

### 7. Response Synthesizer

**Purpose**: Combines all agent outputs into natural conversational responses.

**Input**: SynthesizerInput with aggregated updates from all agents
**Process**:
1. Gathers structured data from concurrent agent executions
2. Grounds response in user's latest message
3. Generates natural reply acknowledging changes and providing focus guidance

## State Management

### Global State

The system uses a TypedDict GlobalState combining messages and synthesizer inputs:

```python
class GlobalState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    synth_input: Annotated[SynthesizerInput, merge_synth_inputs]
    router_decision: Optional[RouterDecision]
```

### Synthesizer Input Merging

Custom reducer merges concurrent agent outputs:

```python
def merge_synth_inputs(left: SynthesizerInput, right: SynthesizerInput) -> SynthesizerInput:
    # Combines task_ids, event_ids, profile changes, etc.
    # Handles concurrent updates from parallel agents
```

## Memory Architecture

### LangGraph Store

**Current Implementation**: InMemoryStore (ephemeral)
**Planned**: SQLite/PostgreSQL for persistence

**Namespaces**:
- `("profile", user_id)`: User profile data
- `("todo", user_id)`: Task list
- `("event", user_id)`: Scheduled events
- `("instructions", user_id)`: User preferences

### Trustcall Integration

Uses Trustcall for structured data extraction from conversations:
- **Tools**: Pydantic models as callable tools
- **Enable Inserts**: Allows creation of new items
- **Memory Context**: Provides existing items for updates

## Graph Flow

### Node Definitions
```python
builder.add_node("conversation_agent", conversation_agent)
builder.add_node("update_todos", update_todos)
builder.add_node("update_events", update_events)
builder.add_node("update_profile", update_profile)
builder.add_node("update_instructions", update_instructions)
builder.add_node("focus_coach", focus_coach)
builder.add_node("response_synthesizer", response_synthesizer)
```

### Edges
```python
builder.add_edge(START, "conversation_agent")
builder.add_conditional_edges("conversation_agent", route_conversation_agent, path_map)
# All agents converge to synthesizer
builder.add_edge("response_synthesizer", END)
```

### Conditional Routing

The `route_conversation_agent` function uses Send objects for dynamic fan-out:

```python
def route_conversation_agent(state: GlobalState):
    decision = state.get("router_decision")
    if decision.disposition == "dispatch":
        return [Send(node_name, state) for target in decision.targets]
    else:
        return END
```

## Configuration

### RunnableConfig
- **user_id**: Unique user identifier for memory isolation
- **timezone**: User's timezone for scheduling

### LangGraph Config
```json
{
  "graphs": {
    "summit": "./src/graph/agent.py:graph"
  },
  "env": "./.env",
  "python_version": "3.13",
  "dependencies": ["."]
}
```

## Key Design Patterns

### LLM-First Routing
- Conversation Agent uses LLM to decide actions rather than rule-based parsing
- Provides flexibility for natural language understanding

### Concurrent Execution
- Multiple agents run in parallel when needed
- State merging handles concurrent updates

### Memory-First Design
- All agents have access to persistent memory
- Context-aware responses based on user history

### Trustcall for Structure
- LLM generates structured data via tool calls
- Ensures data consistency and validation

## Dependencies

**Core Framework**:
- langgraph: Graph orchestration
- langchain-core: LLM integration
- langchain-openai: OpenAI API client

**Memory & Persistence**:
- langgraph-store-base: Memory interface
- langgraph-checkpoint-memory: In-memory checkpointing
- psycopg2-binary: PostgreSQL driver (for future persistence)

**Tools**:
- trustcall: Structured data extraction
- pydantic: Data validation

## Future Architecture Considerations

### Scalability
- **Database Migration**: Move from InMemoryStore to persistent storage
- **User Isolation**: Multi-tenant memory management
- **Caching**: Response caching for repeated queries

### Advanced Features
- **Conflict Resolution**: AI-powered scheduling conflict resolution
- **Learning**: Adaptive suggestions based on user patterns
- **Integration**: Calendar API connections (Google Calendar, Outlook)

### Performance
- **Streaming**: Real-time partial responses
- **Batch Processing**: Handle multiple tasks efficiently
- **Async Execution**: Non-blocking agent operations

This architecture provides a robust foundation for conversational task management while maintaining extensibility for future enhancements.