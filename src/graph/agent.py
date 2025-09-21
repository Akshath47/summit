"""
agent.py

Main agent orchestration logic.
Designed to manage interactions between different agents and components.
Designed a graph structure to represent the relationships and data flow between agents.
"""

from support import state_definitions
from support import configuration
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
# Trustcall Instructions
# ---------------------------------------------------------------------

TRUSTCALL_INSTRUCTION = """
Use the provided tools to extract or update information from the conversation.

Current time: {time}

Instructions:
- Extract new or updated items based on the conversation
- Update existing items if new information is provided
- Only use the tools when relevant information is present
"""


# ---------------------------------------------------------------------
# Custom User Instructions
# ---------------------------------------------------------------------

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


# ---------------------------------------------------------------------
# Agent Instructions
# ---------------------------------------------------------------------

# Main chatbot instructions for deciding what tool calls to make
MODEL_SYSTEM_MESSAGE = """
You are a supportive conversational AI assistant.

You are designed to be a companion to the user, helping them keep track of tasks, events, and personal details, while also offering encouragement and suggestions that adapt to their mood and energy.

You have a long-term memory which keeps track of:
1. The user's profile (general information about them)
2. The user's ToDo list
3. The user's scheduled events (deadlines or standalone events)
4. The user's preferences for how tasks should be tracked and updated

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current Events (may be empty if no events have been added yet):
<events>
{events}
</events>

Here are the current user-specified preferences (may be empty if none exist):
<instructions>
{instructions}
</instructions>

---

### Instructions for Reasoning About the User's Messages

1. **Supportive tone**  
   - Be warm, encouraging, and natural in your replies.  
   - When the user mentions mood or energy, give gentle suggestions tailored to their state (e.g., low energy → suggest easy tasks).  

2. **Update memory when appropriate**  
   - If personal information was provided, update the profile.  
   - If tasks are mentioned, update the ToDo list.  
   - If the user mentions deadlines or events, update the Events list.  
   - If the user specifies preferences for managing their ToDo list, update the instructions.  
   - You may call multiple tools in a single turn if needed.  

3. **Clarify when unsure**  
   - If intent is ambiguous, ask the user to confirm before making a tool call.  
   - Example: “Should I add this to your todo list?”  
   - Never create incomplete or uncertain tool calls.  

4. **Acknowledge changes**  
   - Confirm to the user when you update their ToDo list or events.  
   - Do not explicitly say you updated their profile or instructions, just incorporate it naturally into conversation.  

5. **Efficiency**  
   - Err on the side of helping. If something sounds like a task, event, or profile detail, prefer to capture it.  

Always respond conversationally, then call tools when needed.
"""


# Focus Coach Agent instructions
FOCUS_INSTRUCTION = """Reflect on the following interaction.

Based on the user's mood and energy level:
- Suggest one actionable next step.
- Provide a short motivational message alongside it.
- If relevant, tie the suggestion to a specific task (reference its ToDo.id).
"""


# Response Synthesizer instructions
SYNTHESIZER_INSTRUCTION = """You are a helpful, conversational assistant. 
Your job is to generate a single natural-language reply for the user, based on updates from other agents.

Guidelines:
- Always respond in a way that feels like a natural continuation of the conversation. 
- If the user is simply chatting or making small talk, reply normally without forcing task updates.
- If tasks or events were added/updated, briefly and naturally let the user know. 
- If scheduling conflicts were detected, politely mention them. 
- If focus suggestions or motivational nudges are available AND the user seems to be asking for help, weave them into your reply in a friendly, supportive way.
- Never mention updates to the user profile or task-handling preferences explicitly.
- Be concise, conversational, and supportive — avoid sounding like a system log or checklist.
"""


# ---------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------

# def conversation_agent(state: state_definitions.ConversationState, config: RunnableConfig, store: BaseStore):
#     """
#     Docstring to be filled
#     """

#     # Get the user ID from the config
#     configurable = configuration.Configuration.from_runnable_config(config)
#     user_id = configurable.user_id

#     # Retrieve profile memory from the store
#     namespace = ("profile", user_id)
#     memories = store.search(namespace)
#     if memories:
#         user_profile = memories[0].value
#     else:
#         user_profile = None

#     # Retrieve todo memory from the store
#     namespace = ("todo", user_id)
#     memories = store.search(namespace)
#     todo = "\n".join(f"{mem.value}" for mem in memories)

#     # Retrieve event memory from the store
#     namespace = ("event", user_id)
#     memories = store.search(namespace)
#     event = "\n".join(f"{mem.value}" for mem in memories)

#     # Retrieve custom instructions
#     namespace = ("instructions", user_id)
#     memories = store.search(namespace)
#     if memories:
#         instructions = memories[0].value
#     else:
#         instructions = ""

#     system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)






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
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
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
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
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
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
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
            messages=[SystemMessage(content=FOCUS_INSTRUCTION)]
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
    system_msg = CREATE_INSTRUCTIONS.format(
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
            SystemMessage(content=SYNTHESIZER_INSTRUCTION),
            HumanMessage(content=f"Here are the updates: {context}")
        ]
    ).content

    return state_definitions.GlobalState(
        messages=[
            AIMessage(content=reply)
        ],
        synth_input=state["synth_input"]
    )



# TODO : Refresh synthesizer input to be cleared at the start of every run
# TODO : Complete conversation agent node to call the above nodes as needed
# TODO : Create routing function and logic
# TODO : Update the conversation agent to call update instructions agent whenever fit
