"""
state_definitions.py

Defines the states and tool schemas for the multi-agent LangGraph workflow.

- ConversationState: main message-passing state (used by the Conversation Agent).
- Task, Profile, and Scheduler structures are callable tools and persisted in memory store(database for persistance).
- The other schemas are agent specific input and output schemas 
"""

from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Annotated
from datetime import datetime
import uuid
from langchain_core.messages import BaseMessage
from typing import TypedDict


# ---------------------------------------------------------------------
# Conversation Agent
# ---------------------------------------------------------------------

class ConversationState(MessagesState):
    """
    Main conversation state.
    Inherits from LangGraph's MessagesState, which handles message history.
    """
    pass


# ---------------------------------------------------------------------
# Task Manager Agent (ToDo)
# [TOOL SCHEMA]
# ---------------------------------------------------------------------

class ToDo(BaseModel):
    """A task object stored persistently in DB (via tool calls)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task ID")
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[str] = Field(
        description="Estimated time to complete the task (hours:minutes).",
        default=None
    )
    deadline: Optional[datetime] = Field(
        description="Deadline if applicable.",
        default=None
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="Priority level of the task.",
        default="medium"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Perceived difficulty of the task.",
        default="medium"
    )
    solutions: List[str] = Field(
        description="List of actionable ideas/steps for completing the task.",
        min_length=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task.",
        default="not started"
    )


# ---------------------------------------------------------------------
# Scheduler Agent
# [TOOL SCHEMA]
# ---------------------------------------------------------------------

class Event(BaseModel):
    """
    A scheduled item that can represent either:
    - A deadline for a task (linked via task_id), OR
    - A standalone event (lecture, meeting, etc.)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event ID")
    color: str = Field(description="Color code for urgency/category (e.g., red, green, blue)")
    location: Optional[str] = None
    notes: Optional[str] = None

    # Core details (fit both events and deadlines)
    title: str = Field(description="Name/title of the event or deadline")
    time: datetime = Field(description="When the event starts or when the deadline is")

    # Link to task if this event is a deadline
    task_id: Optional[str] = Field(
        default=None,
        description="Reference to ToDo.id if this event corresponds to a task deadline"
    )


# ---------------------------------------------------------------------
# Profile Update Agent
# [TOOL SCHEMA]
# ---------------------------------------------------------------------

class Profile(BaseModel):
    """User profile information stored persistently."""
    name: Optional[str] = Field(description="User's name", default=None)
    location: Optional[str] = Field(description="User's location", default=None)
    job: Optional[str] = Field(description="User's job/role", default=None)
    college: Optional[str] = Field(description="User's college, if a student", default=None)
    course: Optional[str] = Field(description="User's field of study, if a student", default=None)
    interests: List[str] = Field(description="User interests", default_factory=list)


# ---------------------------------------------------------------------
# Custom Instructions
# [TOOL SCHEMA]
# ---------------------------------------------------------------------

class Instructions(BaseModel):
    """
    User-specific preferences for how tasks should be updated.
    Represented as a simple list of instruction strings.
    """
    items: List[str] = Field(
        default_factory=list,
        description="List of user preferences as natural language instructions"
    )


# ---------------------------------------------------------------------
# Focus Coach Agent (Mood/Energy → Suggestions)
# [STATE SCHEMA]
# ---------------------------------------------------------------------

class FocusContext(BaseModel):
    """State passed to the Focus Coach agent for reasoning."""
    mood: Optional[str] = Field(description="User's current mood", default=None)
    energy: Optional[str] = Field(description="Energy level (low/medium/high)", default=None)

class FocusSuggestion(BaseModel):
    """
    Output from the Focus Coach agent.

    Suggests an action and provides motivation.
    Can reference a task_id if the suggestion is tied to a specific task.
    """
    suggestion: str
    motivation: str
    task_id: Optional[str] = Field(
        default=None,
        description="Reference to a ToDo.id if the suggestion relates to a specific task"
    )


# ---------------------------------------------------------------------
# Response Synthesizer
# [STATE SCHEMA]
# ---------------------------------------------------------------------

class SynthesizerInput(BaseModel):
    """
    Aggregated info from all agents that the Response Synthesizer
    will turn into a natural reply string.
    """
    updated_task_ids: List[str] = Field(default_factory=list)   # Which tasks got created/updated
    event_ids: List[str] = Field(default_factory=list)          # Which events were created/updated
    conflict_ids: List[str] = Field(default_factory=list)       # Events in conflict (if any)
    profile_changes: Optional[dict] = None 
    suggestion: Optional[str] = None   # Next action
    motivation: Optional[str] = None   


def merge_synth_inputs(left: SynthesizerInput, right: SynthesizerInput) -> SynthesizerInput:
    """
    Custom reducer to merge multiple SynthesizerInput updates from concurrent agents.
    """
    return SynthesizerInput(
        updated_task_ids=left.updated_task_ids + right.updated_task_ids,
        event_ids=left.event_ids + right.event_ids,
        conflict_ids=left.conflict_ids + right.conflict_ids,
        profile_changes=right.profile_changes or left.profile_changes,
        suggestion=right.suggestion or left.suggestion,
        motivation=right.motivation or left.motivation,
    )


# ---------------------------------------------------------------------
# Router Decision (LLM-first routing contract)
# ---------------------------------------------------------------------

class RouterDecision(BaseModel):
    """
    Decision returned by the Conversation Agent's router LLM.

    - disposition: how to handle the turn
      * "reply"     -> return a friendly conversational reply, no dispatch
      * "clarify"   -> ask 1 concise question to fill a critical gap, no dispatch
      * "dispatch"  -> fan-out to one or more downstream agents; a short ack is okay here
    - targets: which agents to dispatch when disposition == "dispatch"
    - conversational_reply: friendly message to send immediately
      * for "reply" / "clarify": the actual reply/question
      * for "dispatch": a short acknowledgment like "On it — working on that now."
    - rationale: optional brief reasoning (for trace/debug)
    """
    disposition: Literal["reply", "clarify", "dispatch"]
    targets: List[Literal["todo", "event", "profile", "instructions", "focus"]] = Field(default_factory=list)
    conversational_reply: str = Field(
        ...,
        description="Friendly message to send to the user. For dispatch, use a brief acknowledgment."
    )


# ---------------------------------------------------------------------
# Global State combining Messages and SynthesizerInput
# ---------------------------------------------------------------------

class GlobalState(TypedDict):
    """
    A global state that includes both conversation messages and synthesizer inputs.
    This allows nodes to update both states simultaneously.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    synth_input: Annotated[SynthesizerInput, merge_synth_inputs]
    router_decision: Optional["RouterDecision"]
