"""
agent.py

Main agent orchestration logic.
Designed to manage interactions between different agents and components.
Designed a graph structure to represent the relationships and data flow between agents.
"""

from support import state_definitions
from support import configuration
from support import prompts
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from datetime import datetime
from trustcall import create_extractor

from langchain_openai import ChatOpenAI

from langgraph.types import Send
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, START, StateGraph


# ---------------------------------------------------------------------
# Initialize model and tools
# ---------------------------------------------------------------------

# LLM setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Memory management
store = InMemoryStore()
checkpointer = MemorySaver()
# TODO : Implement persistent storage with SQLite or Postgres


# ---------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------

def conversation_agent(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Conversation Agent node.
    Parses user message, decides which agents to invoke, and clears synth_input for fresh run.
    """

    # Clear synthesizer input at the start of every run
    synth_input = state_definitions.SynthesizerInput()

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve todo memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve event memory from the store
    namespace = ("event", user_id)
    memories = store.search(namespace)
    event = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    # Build router prompt
    router_system = SystemMessage(
        content=prompts.MODEL_SYSTEM_MESSAGE.format(
            user_profile=user_profile or "",
            todo=todo or "",
            events=event or "",
            instructions=instructions or "",
        )
    )
    router_instruction = SystemMessage(content=prompts.ROUTER_INSTRUCTION)
    last_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")

    # Ask LLM for routing decision (LLM-first policy)
    decision_model = llm.with_structured_output(state_definitions.RouterDecision)
    decision = decision_model.invoke([router_system, router_instruction, last_message])

    # Map targets to node names
    target_to_node = {
        "todo": "update_todos",
        "event": "update_events",
        "profile": "update_profile",
        "instructions": "update_instructions",
        "focus": "focus_coach",
    }

    # Prepare fan-out sends if dispatching
    if getattr(decision, "disposition", None) == "dispatch":
        sends = []
        for t in getattr(decision, "targets", []) or []:
            node_name = target_to_node.get(t)
            if node_name:
                sends.append(
                    Send(
                        node_name,
                        state_definitions.GlobalState(
                            messages=state["messages"],
                            synth_input=synth_input
                        ),
                    )
                )

        # Return short acknowledgment now; final reply comes from response_synthesizer
        ack = getattr(decision, "conversational_reply", None) or "On it — working on that now."
        return {
            "messages": [AIMessage(content=ack)],
            "synth_input": synth_input,
        } | {"__sends__": sends}

    # Reply or Clarify: send friendly message, no dispatch
    reply_text = getattr(decision, "conversational_reply", None) or "Okay."
    return {
        "messages": [AIMessage(content=reply_text)],
        "synth_input": synth_input,
    } | {"__sends__": []}


def update_todos(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Task Manager Agent node.
    Reflects on conversation, updates ToDos in persistent memory, and confirms changes.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define namespace for todos
    namespace = ("todo", user_id)

    # Retrieve existing todos (for context)
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "ToDo", item.value) for item in existing_items] if existing_items else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = prompts.TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Create the todo extractor
    todo_extractor = create_extractor(
        llm,
        tools=[state_definitions.ToDo],
        tool_choice="ToDo",
        enable_inserts=True,
    )
    
    # Invoke the extractor
    result = todo_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Collect updated task IDs for synthesizer
    updated_task_ids = [rmeta.get("json_doc_id", str(uuid.uuid4())) for _, rmeta in zip(result["responses"], result["response_metadata"])]

    # Summaries by counts
    insert_count = sum(1 for m in result["response_metadata"] if m.get("is_insert", False))
    update_count = len(result["response_metadata"]) - insert_count

    lines = []
    if insert_count:
        lines.append(f"Added {insert_count} task(s).")
    if update_count:
        lines.append(f"Updated {update_count} task(s).")
    summary_text = "\n".join(lines) if lines else "No changes made."

    # Confirmation back to Conversation Agent
    last_message = state["messages"][-1]
    tool_call_id = getattr(last_message, 'tool_calls', [{}])[0].get('id', 'default_id')

    # Merge synthesizer input
    new_synth_input = state["synth_input"].model_copy(update={"updated_task_ids": state["synth_input"].updated_task_ids + updated_task_ids})

    return state_definitions.GlobalState(
        messages=[
            ToolMessage(
                content=summary_text, 
                tool_call_id=tool_call_id
            )
        ],
        synth_input=new_synth_input
    )


def update_profile(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Profile Update Agent node.
    Reflects on conversation, updates Profile in persistent memory.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define namespace for profile
    namespace = ("profile", user_id)

    # Retrieve existing profile (for context)
    # There is only one profile but we are using plural naming for consistency across trustcall calls
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "Profile", item.value) for item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = prompts.TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Create the profile extractor
    profile_extractor = create_extractor(
        llm,
        tools=[state_definitions.Profile],
        tool_choice="Profile",
        enable_inserts=True,
    )

    # Invoke the extractor
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save save the memories from Trustcall to the store
    summaries = []
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
        summaries.append("Profile updated.")

    # Collect profile changes for synthesizer
    profile_changes = result["responses"][0].model_dump(mode="json") if result["responses"] else None

    summary_text = "\n".join(summaries) if summaries else "No profile changes."

    # Confirmation back to Conversation Agent
    last_message = state["messages"][-1]
    tool_call_id = getattr(last_message, "tool_calls", [{}])[0].get("id", "default_id")

    # Merge synthesizer input
    new_synth_input = state["synth_input"].model_copy(update={"profile_changes": profile_changes})

    return state_definitions.GlobalState(
        messages=[
            ToolMessage(
                content=summary_text, 
                tool_call_id=tool_call_id
            )
        ],
        synth_input=new_synth_input
    )


def update_events(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Event Manager Agent node.
    Reflects on conversation, updates Events in persistent memory.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define namespace for events
    namespace = ("event", user_id)

    # Retrieve existing events (for context)
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "Event", item.value) for item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = prompts.TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Create the profile extractor
    event_extractor = create_extractor(
        llm,
        tools=[state_definitions.Event],
        tool_choice="Event",
        enable_inserts=True,
    )

    # Invoke the extractor 
    result = event_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save new/updated events
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Collect updated event IDs for synthesizer
    updated_event_ids = [rmeta.get("json_doc_id", str(uuid.uuid4())) for _, rmeta in zip(result["responses"], result["response_metadata"])]

    # Build a light summary (just counts)
    insert_count = sum(1 for m in result["response_metadata"] if m.get("is_insert", False))
    update_count = len(result["response_metadata"]) - insert_count

    lines = []
    if insert_count:
        lines.append(f"Added {insert_count} event(s).")
    if update_count:
        lines.append(f"Updated {update_count} event(s).")
    summary_text = "\n".join(lines) if lines else "No event changes."

    # Confirmation back to Conversation Agent
    last_message = state["messages"][-1]
    tool_call_id = getattr(last_message, "tool_calls", [{}])[0].get("id", "default_id")

    # Merge synthesizer input
    new_synth_input = state["synth_input"].model_copy(update={"event_ids": state["synth_input"].event_ids + updated_event_ids})

    return state_definitions.GlobalState(
        messages=[
            ToolMessage(content=summary_text, tool_call_id=tool_call_id)
        ],
        synth_input=new_synth_input
    )


def focus_coach(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Focus Coach Agent node.
    Uses mood + energy signals to produce a FocusSuggestion,
    which is later consumed by the Response Synthesizer.
    """

    # Get the user ID from the config (not really needed, but for consistency sake)
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Prepare messages
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=prompts.FOCUS_INSTRUCTION)]
            + state["messages"][:-1]
        )
    )

    # Create the FocusSuggestion extractor
    focus_extractor = create_extractor(
        llm,
        tools=[state_definitions.FocusSuggestion],
        tool_choice="FocusSuggestion",
        enable_inserts=True,
    )

    # Invoke the extractor
    result = focus_extractor.invoke({"messages": updated_messages, "existing": None})

    # Collect structured output for the synthesizer
    if result["responses"]:
        r = result["responses"][0]

        if hasattr(r, "model_dump"):
            data = r.model_dump()
        elif isinstance(r, dict):
            data = r
        else:
            data = {}

        content = {
            "suggestion": data.get("suggestion"),
            "motivation": data.get("motivation"),
        }
    else:
        content = {"suggestion": None, "motivation": None}

    # Ensure required list fields are always lists (not None)
    new_synth_input = state["synth_input"].model_copy(update=content)

    return state_definitions.GlobalState(
        messages=[],
        synth_input=new_synth_input
    )


def update_instructions(state: state_definitions.ConversationState, config: RunnableConfig, store: BaseStore):
    """
    Instructions Update Agent node.
    Updates user preferences for how ToDos should be added/handled.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define namespace for instructions
    namespace = ("instructions", user_id)

    # Retrieve existing instructions (list of strings)
    existing_memory = store.get(namespace, "user_instructions")
    current_instructions = existing_memory.value if existing_memory else []

    # Format system prompt
    system_msg = prompts.CREATE_INSTRUCTIONS.format(
        current_instructions="\n".join(current_instructions) if current_instructions else "None"
    )

    # Bind instructions schema to the LLM
    instructions_model = llm.with_structured_output(state_definitions.Instructions)

    # Ask the model for updated instructions (schema-bound)
    result = instructions_model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [HumanMessage(content="Please update the instructions based on this conversation")]
    )

    # Save updated list directly
    store.put(namespace, "user_instructions", {"items": result.items})

    # Confirm back to Conversation Agent
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_call_id = tool_calls[0]["id"] if tool_calls else "default_id"
    return {
        "messages": [
            {
                "role": "tool",
                "content": "Updated your preferences for managing tasks.",
                "tool_call_id": tool_call_id,
            }
        ]
    }


def response_synthesizer(
    state: state_definitions.GlobalState,
    config: RunnableConfig,
    store: BaseStore,
):
    """Use the LLM to synthesize a natural-language response for the user."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    def load_records(namespace_key, keys):
        namespace = (namespace_key, user_id)
        records = []
        for k in keys or []:
            mem = store.get(namespace, k)
            if mem:
                records.append(mem.value)
        return records

    # Gather structured updates
    todos = load_records("todo", state["synth_input"].updated_task_ids)
    events = load_records("event", state["synth_input"].event_ids)
    conflicts = state["synth_input"].conflict_ids or []

    # Focus agent outputs
    focus = {
        "suggestion": state["synth_input"].suggestion,
        "motivation": state["synth_input"].motivation,
    }

    # Build input context
    context = {
        "tasks": [t.get("task", "a task") for t in todos],
        "events": [
            {"title": e.get("title", "an event"), "time": e.get("time")}
            for e in events
        ],
        "conflicts": len(conflicts),
        "focus": focus,
    }

    # Call LLM
    reply = llm.invoke(
        [
            SystemMessage(content=prompts.SYNTHESIZER_INSTRUCTION),
            HumanMessage(content=f"Here are the updates: {context}")
        ]
    ).content

    return state_definitions.GlobalState(
        messages=[
            AIMessage(content=reply)
        ],
        synth_input=state["synth_input"]
    )


# TODO : Client side streaming of messages to get partial updates on tool calls
