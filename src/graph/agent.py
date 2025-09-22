"""
agent.py

Main agent orchestration logic.
Designed to manage interactions between different agents and components.
Designed a graph structure to represent the relationships and data flow between agents.
"""

from src.support import state_definitions
from src.support import configuration
from src.support import prompts
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from datetime import datetime
from trustcall import create_extractor

from langchain_openai import ChatOpenAI

from langgraph.types import Send, Command
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, START, StateGraph


# ---------------------------------------------------------------------
# Initialize model and tools
# ---------------------------------------------------------------------

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    # DEBUG: Log current synth_input state before clearing
    current_synth = state.get("synth_input")
    if current_synth:
        print(f"[DEBUG] conversation_agent - BEFORE clearing:")
        print(f"  updated_task_ids: {len(current_synth.updated_task_ids)} items")
        print(f"  event_ids: {len(current_synth.event_ids)} items")
        print(f"  conflict_ids: {len(current_synth.conflict_ids)} items")
    
    # Clear synthesizer input at the start of every run
    synth_input = state_definitions.SynthesizerInput()
    print(f"[DEBUG] conversation_agent - AFTER clearing: created fresh SynthesizerInput")

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
    # Smart context detection: check if previous turn was a clarification/reply
    messages_to_use = []
    if (len(state["messages"]) >= 3 and
        isinstance(state["messages"][-2], AIMessage) and
        getattr(state.get("router_decision"), "disposition", None) in {"clarify"}):
        # Include last 3 messages: original user query, AI clarification, user response
        messages_to_use = state["messages"][-3:]
    else:
        # Normal case: just the latest message
        messages_to_use = [state["messages"][-1]] if state["messages"] else [HumanMessage(content="")]

    # Ask LLM for routing decision with appropriate context (LLM-first policy)
    decision_model = llm.with_structured_output(state_definitions.RouterDecision)
    decision = decision_model.invoke([router_system, router_instruction] + messages_to_use)

    # Map targets to node names
    target_to_node = {
        "todo": "update_todos",
        "event": "update_events",
        "profile": "update_profile",
        "instructions": "update_instructions",
        "focus": "focus_coach",
    }

    # Store the decision in state for the conditional edge to use
    # Return short acknowledgment for dispatch, or full reply for reply/clarify
    if getattr(decision, "disposition", None) == "dispatch":
        ack = getattr(decision, "conversational_reply", None) or "On it — working on that now."
        return {
            "messages": [AIMessage(content=ack)],
            "synth_input": synth_input,
            "router_decision": decision,  # Store decision for conditional edge
        }
    else:
        # Reply or Clarify: send friendly message, no dispatch
        reply_text = getattr(decision, "conversational_reply", None) or "Okay."
        return {
            "messages": [AIMessage(content=reply_text)],
            "synth_input": synth_input,
            "router_decision": decision,  # Store decision for conditional edge
        }


def update_todos(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Task Manager Agent node.
    Reflects on conversation, updates ToDos in persistent memory, and confirms changes.
    """

    # DEBUG: Log received synth_input state
    current_synth = state.get("synth_input")
    if current_synth:
        print(f"[DEBUG] update_todos - RECEIVED state:")
        print(f"  updated_task_ids: {len(current_synth.updated_task_ids)} items: {current_synth.updated_task_ids}")
        print(f"  event_ids: {len(current_synth.event_ids)} items: {current_synth.event_ids}")
        print(f"  conflict_ids: {len(current_synth.conflict_ids)} items: {current_synth.conflict_ids}")

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Validate user_id is not empty to prevent InvalidNamespaceError
    if not user_id or user_id.strip() == "":
        user_id = "default-user"

    # Define namespace for todos
    namespace = ("todo", user_id)

    # Retrieve existing todos (for context)
    existing_items = store.search(namespace)
    existing_memories = (
        [(item.key, "ToDo", item.value) for item in existing_items] if existing_items else None
    )

    # Merge the chat history and the instruction (ToDo-specific, ensures 'solutions' populated)
    TODO_TRUSTCALL_INSTRUCTION_FORMATTED = prompts.TODO_TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    
    # Filter out ToolMessages that don't have corresponding tool_calls to prevent API errors
    filtered_messages = []
    for i, msg in enumerate(state["messages"][:-1]):
        if hasattr(msg, 'type') and msg.type == 'tool':
            # Check if previous message has tool_calls
            if i > 0:
                prev_msg = state["messages"][i-1]
                tool_calls = getattr(prev_msg, 'tool_calls', None)
                if tool_calls:
                    filtered_messages.append(msg)
            # Skip ToolMessages without corresponding tool_calls
        else:
            filtered_messages.append(msg)
    
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TODO_TRUSTCALL_INSTRUCTION_FORMATTED)]
            + filtered_messages
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

    # Ensure 'solutions' is populated for created/updated tasks as a fallback (operate on payload dicts, not model attrs)
    import json
    normalized_payloads = []
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        # Normalize to dict payload
        if hasattr(r, "model_dump"):
            payload = r.model_dump(mode="json")
        elif isinstance(r, dict):
            payload = r
        else:
            payload = {}
        solutions = payload.get("solutions") or []
        if not solutions:
            task_text = payload.get("task")
            steps: list[str] = []
            if task_text:
                # Ask LLM for concrete micro-steps (prefer JSON array)
                step_prompt = (
                    "Generate 3-6 concrete, bite-sized, action-oriented micro-steps (5-20 minutes each) "
                    f"to progress the task: '{task_text}'. Return ONLY a JSON array of strings."
                )
                raw = llm.invoke([SystemMessage(content=step_prompt)]).content
                steps_list: list = []
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            steps_list = parsed
                    except Exception:
                        # Fallback: parse line-by-line if not valid JSON
                        steps_list = [s.strip() for s in raw.splitlines() if s and s.strip()]
                elif isinstance(raw, list):
                    steps_list = [str(x) for x in raw]
                steps = [str(s).strip().lstrip("-").strip() for s in steps_list if str(s).strip()][:6]
            if not steps:
                steps = [
                    "Outline the first 3 bullet points",
                    "Draft a rough version for 10 minutes",
                    "List blockers or missing info",
                ]
            payload["solutions"] = steps
        normalized_payloads.append((payload, rmeta))

    # Save the memories from Trustcall to the store (with ensured 'solutions')
    for payload, rmeta in normalized_payloads:
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            payload,
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
    tool_calls = getattr(last_message, 'tool_calls', None) or []
    tool_call_id = tool_calls[0].get('id', 'default_id') if tool_calls else 'default_id'

    # Create fresh synthesizer input with only new task IDs (merge function will combine with other nodes)
    new_synth_input = state_definitions.SynthesizerInput(updated_task_ids=updated_task_ids)
    
    # DEBUG: Log what we're returning
    print(f"[DEBUG] update_todos - RETURNING:")
    print(f"  updated_task_ids: {len(new_synth_input.updated_task_ids)} items: {new_synth_input.updated_task_ids}")

    return {
        "messages": [
            ToolMessage(
                content=summary_text,
                tool_call_id=tool_call_id
            )
        ],
        "synth_input": new_synth_input
    }


def update_profile(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Profile Update Agent node.
    Reflects on conversation, updates Profile in persistent memory.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Validate user_id is not empty to prevent InvalidNamespaceError
    if not user_id or user_id.strip() == "":
        user_id = "default-user"

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
    
    # Filter out ToolMessages that don't have corresponding tool_calls to prevent API errors
    filtered_messages = []
    for i, msg in enumerate(state["messages"][:-1]):
        if hasattr(msg, 'type') and msg.type == 'tool':
            # Check if previous message has tool_calls
            if i > 0:
                prev_msg = state["messages"][i-1]
                tool_calls = getattr(prev_msg, 'tool_calls', None)
                if tool_calls:
                    filtered_messages.append(msg)
            # Skip ToolMessages without corresponding tool_calls
        else:
            filtered_messages.append(msg)
    
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + filtered_messages
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
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_call_id = tool_calls[0].get("id", "default_id") if tool_calls else "default_id"

    # Create fresh synthesizer input with only profile changes (merge function will combine with other nodes)
    new_synth_input = state_definitions.SynthesizerInput(profile_changes=profile_changes)

    return {
        "messages": [
            ToolMessage(
                content=summary_text,
                tool_call_id=tool_call_id
            )
        ],
        "synth_input": new_synth_input
    }


def update_events(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Event Manager Agent node.
    Reflects on conversation, updates Events in persistent memory.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Validate user_id is not empty to prevent InvalidNamespaceError
    if not user_id or user_id.strip() == "":
        user_id = "default-user"

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
    
    # Filter out ToolMessages that don't have corresponding tool_calls to prevent API errors
    filtered_messages = []
    for i, msg in enumerate(state["messages"][:-1]):
        if hasattr(msg, 'type') and msg.type == 'tool':
            # Check if previous message has tool_calls
            if i > 0:
                prev_msg = state["messages"][i-1]
                tool_calls = getattr(prev_msg, 'tool_calls', None)
                if tool_calls:
                    filtered_messages.append(msg)
            # Skip ToolMessages without corresponding tool_calls
        else:
            filtered_messages.append(msg)
    
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + filtered_messages
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
    tool_calls = getattr(last_message, 'tool_calls', None) or []
    tool_call_id = tool_calls[0].get('id', 'default_id') if tool_calls else 'default_id'

    # Create fresh synthesizer input with only new event IDs (merge function will combine with other nodes)
    new_synth_input = state_definitions.SynthesizerInput(event_ids=updated_event_ids)

    return {
        "messages": [
            ToolMessage(content=summary_text, tool_call_id=tool_call_id)
        ],
        "synth_input": new_synth_input
    }


def focus_coach(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Focus Coach Agent node.
    Uses mood + energy signals to produce a FocusSuggestion,
    which is later consumed by the Response Synthesizer.
    """

    # Get the user ID from the config (not really needed, but for consistency sake)
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Load current ToDos and Events for grounding
    todo_items = store.search(("todo", user_id)) or []
    event_items = store.search(("event", user_id)) or []

    # Build compact catalogs for tasks/events
    def _safe_str(val):
        try:
            return val.isoformat() if hasattr(val, "isoformat") else str(val)
        except Exception:
            return str(val)

    todo_lines = []
    for item in todo_items:
        v = item.value if isinstance(item.value, dict) else {}
        tid = v.get("id")
        task = v.get("task")
        priority = v.get("priority")
        difficulty = v.get("difficulty")
        ttc = v.get("time_to_complete")
        deadline = v.get("deadline")
        status = v.get("status")
        solutions = v.get("solutions") or []
        solutions_preview = "; ".join((s or "") for s in solutions[:3])
        todo_lines.append(
            f"- id={tid}; task={task}; priority={priority}; difficulty={difficulty}; ttc={ttc}; deadline={_safe_str(deadline) if deadline else ''}; status={status}; solutions={solutions_preview}"
        )

    event_lines = []
    for item in event_items:
        v = item.value if isinstance(item.value, dict) else {}
        eid = v.get("id")
        title = v.get("title")
        time_val = v.get("time")
        task_id = v.get("task_id")
        event_lines.append(
            f"- id={eid}; title={title}; time={_safe_str(time_val) if time_val else ''}; task_id={task_id}"
        )

    todo_catalog = "\n".join(todo_lines) if todo_lines else "None"
    event_catalog = "\n".join(event_lines) if event_lines else "None"

    # Format Focus system prompt with catalogs
    focus_system = prompts.FOCUS_INSTRUCTION.format(
        todo_catalog=todo_catalog,
        event_catalog=event_catalog
    )

    # Filter out ToolMessages that don't have corresponding tool_calls to prevent API errors
    filtered_messages = []
    for i, msg in enumerate(state["messages"][:-1]):
        if hasattr(msg, 'type') and msg.type == 'tool':
            # Check if previous message has tool_calls
            if i > 0:
                prev_msg = state["messages"][i-1]
                tool_calls = getattr(prev_msg, 'tool_calls', None)
                if tool_calls:
                    filtered_messages.append(msg)
            # Skip ToolMessages without corresponding tool_calls
        else:
            filtered_messages.append(msg)

    # Prepare messages with grounded catalogs
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=focus_system)]
            + filtered_messages
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

    # Create fresh synthesizer input with only focus content (merge function will combine with other nodes)
    new_synth_input = state_definitions.SynthesizerInput(
        suggestion=content.get("suggestion"),
        motivation=content.get("motivation")
    )

    return {
        "messages": [],
        "synth_input": new_synth_input
    }


def update_instructions(state: state_definitions.GlobalState, config: RunnableConfig, store: BaseStore):
    """
    Instructions Update Agent node.
    Updates user preferences for how ToDos should be added/handled.
    """

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Validate user_id is not empty to prevent InvalidNamespaceError
    if not user_id or user_id.strip() == "":
        user_id = "default-user"

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
    tool_call_id = tool_calls[0].get("id", "default_id") if tool_calls else "default_id"
    
    return {
        "messages": [
            ToolMessage(
                content="Updated your preferences for managing tasks.",
                tool_call_id=tool_call_id,
            )
        ],
        "synth_input": state_definitions.SynthesizerInput(instructions_updated=True)
    }


def response_synthesizer(
    state: state_definitions.GlobalState,
    config: RunnableConfig,
    store: BaseStore,
):
    """Use the LLM to synthesize a natural-language response for the user."""

    # DEBUG: Log what response_synthesizer receives
    current_synth = state.get("synth_input")
    if current_synth:
        print(f"[DEBUG] response_synthesizer - RECEIVED state:")
        print(f"  updated_task_ids: {len(current_synth.updated_task_ids)} items: {current_synth.updated_task_ids}")
        print(f"  event_ids: {len(current_synth.event_ids)} items: {current_synth.event_ids}")
        print(f"  conflict_ids: {len(current_synth.conflict_ids)} items: {current_synth.conflict_ids}")

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

    # Build input context with better structure
    has_tasks = len(todos) > 0
    has_events = len(events) > 0
    has_profile_update = state["synth_input"].profile_changes is not None
    has_instructions_update = getattr(state["synth_input"], "instructions_updated", False)
    has_focus = focus["suggestion"] is not None
    
    context = {
        "tasks_added": len([t for t in todos if t]) if has_tasks else 0,
        "task_details": [t.get("task", "a task") for t in todos] if has_tasks else [],
        "events_added": len([e for e in events if e]) if has_events else 0,
        "event_details": [{"title": e.get("title", "an event"), "time": e.get("time")} for e in events] if has_events else [],
        "conflicts": len(conflicts),
        "profile_updated": has_profile_update,
        "instructions_updated": has_instructions_update,
        "focus_suggestion": focus["suggestion"] if has_focus else None,
        "focus_motivation": focus["motivation"] if has_focus else None,
    }

    # Build concise context summary for the LLM
    parts = []
    if context["tasks_added"] > 0:
        tasks_preview = ", ".join(context["task_details"][:3])
        parts.append(f"Tasks: {context['tasks_added']} updated ({tasks_preview})")
    if context["events_added"] > 0:
        parts.append(f"Events: {context['events_added']} updated")
    if context["profile_updated"]:
        parts.append("Profile updated")
    if context["instructions_updated"]:
        parts.append("Instructions updated")
    if context["conflicts"] > 0:
        parts.append(f"Conflicts: {context['conflicts']}")
    if context["focus_suggestion"]:
        mot = f" — {context['focus_motivation']}" if context["focus_motivation"] else ""
        parts.append(f"Focus: {context['focus_suggestion']}{mot}")
    context_text = "; ".join(parts) if parts else "No structured updates this turn."
    
    # Find the latest human message to ground the reply
    last_user_content = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_content = m.content
            break
    
    # DEBUG: Log synthesizer grounding inputs
    print(f"[DEBUG] response_synthesizer - context_text: {context_text}")
    print(f"[DEBUG] response_synthesizer - last_user_content: {last_user_content[:200]}{'...' if len(last_user_content) > 200 else ''}")
    
    # Call LLM with grounded prompt
    reply = llm.invoke(
        [
            SystemMessage(content=prompts.SYNTHESIZER_INSTRUCTION),
            SystemMessage(content=f"Context: {context_text}"),
            HumanMessage(content=f"User message: {last_user_content}")
        ]
    ).content

    return {
        "messages": [
            AIMessage(content=reply)
        ],
        "synth_input": state["synth_input"]
    }


# ---------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------

def route_conversation_agent(state: state_definitions.GlobalState):
    """
    Conditional edge function that returns Send objects based on router decision.
    This is where the Send pattern should be implemented according to LangGraph docs.
    """
    decision = state.get("router_decision")
    if not decision:
        return END
    
    # Map targets to node names
    target_to_node = {
        "todo": "update_todos",
        "event": "update_events",
        "profile": "update_profile",
        "instructions": "update_instructions",
        "focus": "focus_coach",
    }
    
    if getattr(decision, "disposition", None) == "dispatch":
        # Return Send objects for each target
        sends = []
        for t in getattr(decision, "targets", []) or []:
            node_name = target_to_node.get(t)
            if node_name:
                # Send the current state to each target node
                sends.append(Send(node_name, state))
        return sends
    else:
        # For reply or clarify, end the conversation
        return END


# ---------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------

# Create the graph + all nodes
builder = StateGraph(state_definitions.GlobalState)

# Define the nodes (sub agents)
builder.add_node("conversation_agent", conversation_agent)
builder.add_node("update_todos", update_todos)
builder.add_node("update_events", update_events)
builder.add_node("update_profile", update_profile)
builder.add_node("update_instructions", update_instructions)
builder.add_node("focus_coach", focus_coach)
builder.add_node("response_synthesizer", response_synthesizer)

# Define the edges (data flow)
builder.add_edge(START, "conversation_agent")

# Use conditional edge with Send objects for dynamic dispatch
# Path map defines the possible paths from the conditional edge
path_map = [
    "update_todos",
    "update_events",
    "update_profile",
    "update_instructions",
    "focus_coach",
    END
]

builder.add_conditional_edges("conversation_agent", route_conversation_agent, path_map)

# All sub agents converge back into synthesizer agent
builder.add_edge("update_todos", "response_synthesizer")
builder.add_edge("update_events", "response_synthesizer")
builder.add_edge("update_profile", "response_synthesizer")
builder.add_edge("update_instructions", "response_synthesizer")
builder.add_edge("focus_coach", "response_synthesizer")

# Response synthesizer ends the flow
builder.add_edge("response_synthesizer", END)

# Compile the graph
# graph = builder.compile(checkpointer=checkpointer, store=store)
graph = builder.compile()

# TODO : Client side streaming of messages to get partial updates on tool calls
