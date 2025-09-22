# Summit

**Summit** is a conversational task and scheduling assistant built with LangGraph. It helps users manage tasks, deadlines, and schedules through natural conversation, while providing personalized motivation and focus suggestions based on mood and energy levels.

> "Just as a summit is the highest point of a journey, this tool is designed to guide students through daily tasks, schedules, and focus suggestions in a way that supports long-term progress."

## Features

- **Conversational Task Management**: Add, update, and complete tasks through natural language
- **Intelligent Scheduling**: Automatically schedule tasks and detect deadline conflicts
- **Profile Learning**: Learns user preferences and personal details over time
- **Focus Coaching**: Provides tailored suggestions based on current mood and energy levels
- **Persistent Memory**: Stores tasks, events, and user data using LangGraph's memory system
- **Multi-Agent Architecture**: Specialized agents handle different aspects of task management

## Architecture

Summit uses a multi-agent LangGraph workflow with the following components:

### Core Agents
- **Conversation Agent**: Parses user messages and routes to appropriate specialized agents
- **Task Manager Agent**: Handles ToDo list creation, updates, and completion
- **Scheduler Agent**: Manages calendar events and deadlines
- **Profile Update Agent**: Learns and stores user profile information
- **Focus Coach Agent**: Provides motivational suggestions based on mood/energy
- **Instructions Update Agent**: Learns user preferences for task handling
- **Response Synthesizer**: Generates natural language responses combining all agent outputs

### Data Flow
```
User Message → Conversation Agent → Specialized Agents → Response Synthesizer → Final Reply
```

## Project Structure

```
summit/
├── src/
│   ├── graph/
│   │   ├── agent.py          # Main LangGraph implementation
│   │   └── __init__.py
│   └── support/
│       ├── configuration.py  # Configuration management
│       ├── prompts.py        # LLM prompts and instructions
│       ├── state_definitions.py  # Pydantic models and state schemas
│       ├── __init__.py
│       ├── requirements.txt  # Python dependencies
│       └── package.json      # (Empty)
├── docs/
│   ├── plan.md              # Original project plan
│   ├── summit.md            # Project naming rationale
│   └── [additional docs]    # Documentation files
├── setup.py                 # Package setup
├── langgraph.json           # LangGraph deployment config
├── .env                     # Environment variables
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd summit
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
# or
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Running the Application

1. Start the LangGraph server:
```bash
langgraph dev
```

2. The application will be available at the configured endpoint.

## Usage

Summit supports natural language interactions for task management:

**Adding Tasks:**
- "Add a task to finish the report by Friday"
- "I need to call the dentist next Tuesday"

**Completing Tasks:**
- "I finished the email to Bob"
- "Mark the report as done"

**Scheduling:**
- "Block time tomorrow afternoon for prep"
- "Schedule the team meeting for Wednesday at 2 PM"

**Focus Support:**
- "I'm exhausted, what should I do now?"
- "Feeling unmotivated, help me decide what's next"

**Profile Updates:**
- "Call me Jay from now on"
- "I'm a computer science student"

## Configuration

The application uses LangGraph's configuration system. Key settings in `langgraph.json`:

- `graphs.summit`: Points to the main graph in `src/graph/agent.py`
- `env`: Environment file path
- `python_version`: Python version requirement

## Current Progress

### ✅ Completed Features
- Multi-agent LangGraph architecture implementation
- Conversation routing with LLM-first policy
- Task management with persistent storage via LangGraph Store
- Event scheduling and calendar integration
- Profile learning and user preferences
- Focus coaching with mood-based suggestions
- Response synthesis for natural conversations
- Basic configuration and prompt management

### 🚧 In Progress / TODO
- **Persistent Storage**: Currently using InMemoryStore; needs migration to SQLite/PostgreSQL
- **Client-side Streaming**: Implement partial message streaming for better UX
- **Deployment**: Production deployment configuration
- **UI/Frontend**: Web interface for interaction
- **Advanced Features**:
  - Conflict detection in scheduling
  - Advanced focus suggestions
  - Integration with external calendars
  - Notification system

### 🐛 Known Issues
- Memory persistence is not persistent across restarts (using InMemoryStore)
- No conflict detection implemented yet
- Limited error handling and validation

## Technology Stack

- **Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o-mini
- **Memory**: LangGraph Store (currently InMemory, planned: SQLite/PostgreSQL)
- **State Management**: TypedDict with custom reducers
- **Tool Calling**: Trustcall for structured data extraction
- **Deployment**: LangGraph CLI and Docker

## Documentation

Additional documentation can be found in the `docs/` directory:
- [Architecture Details](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [Progress Report](docs/progress.md)
- [Project Plan](docs/plan.md)