"""
agent.py

Main agent orchestration logic.
Designed to manage interactions between different agents and components.
Designed a graph structure to represent the relationships and data flow between agents.
"""

from support import state_definitions
from support import configuration
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string, merge_message_runs
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



event_extractor = create_extractor(
    llm,
    tools=[state_definitions.Event],
    tool_choice="Event",
    enable_inserts=True,
)


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


# Custom Instructions
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


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






def update_todos(state: state_definitions.ConversationState, config: RunnableConfig, store: BaseStore):
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
    return {
        "messages": [
            {
                "role": "tool",
                "content": summary_text,
                "tool_call_id": tool_call_id,
            }
        ]
    }


def update_profile(state: state_definitions.ConversationState, config: RunnableConfig, store: BaseStore):
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

    # Run extraction using the global profile_extractor
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

    summary_text = "\n".join(summaries) if summaries else "No profile changes."

    # Confirmation back to Conversation Agent
    last_message = state["messages"][-1]
    tool_call_id = getattr(last_message, "tool_calls", [{}])[0].get("id", "default_id")
    return {
        "messages": [
            {
                "role": "tool",
                "content": summary_text,
                "tool_call_id": tool_call_id,
            }
        ]
    }


