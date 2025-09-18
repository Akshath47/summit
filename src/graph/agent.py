"""
agent.py

Main agent orchestration logic.
Designed to manage interactions between different agents and components.
Designed a graph structure to represent the relationships and data flow between agents.
"""

import state_definitions
import configuration
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SQLiteStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, START, StateGraph


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0) 


# ---------------------------------------------------------------------
# Memory Management
# ---------------------------------------------------------------------
store = SQLiteStore.from_file("memory.db")
checkpointer = SqliteSaver.from_file("checkpoints.db")


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

def conversation_agent(state: state_definitions.ConversationState, config: RunnableConfig, store: BaseStore):
    """
    Docstring to be filled
    """

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

    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)


