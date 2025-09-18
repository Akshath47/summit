"""
state_definitions.py

Defines the states and tool schemas for the multi-agent LangGraph workflow.

- ConversationState: main message-passing state (used by the Conversation Agent).
- Task, Profile, and Scheduler structures are callable tools and persisted in memory store(database for persistance).
- The other schemas are agent specific input and output schemas 
"""

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
import uuid


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
        min_items=1,
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
    id: str = Field("user_instructions", description="Unique key for this user's instruction set")
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
    updated_task_ids: List[str] = []   # Which tasks got created/updated
    event_ids: List[str] = []          # Which events were created/updated
    conflict_ids: List[str] = []       # Events in conflict (if any)
    profile_changes: Optional[dict] = None 
    suggestion: Optional[str] = None   # Next action
    motivation: Optional[str] = None   

class ResponseState(MessagesState):
    """Final response returned to the user."""
    reply: str

