# 📌 Conversational Task & Scheduling Agent (LangGraph)

This project is a **multi-agent LangGraph workflow** that helps users manage tasks, deadlines, and schedules through natural conversation.  
It personalizes responses, maintains a profile of the user, and provides actionable suggestions based on mood/energy and deadlines.  

---

## 🔹 Agents Overview

### 1. Conversation Agent
- Parses the user’s natural message into structured data.  
- Extracts tasks, deadlines, profile updates, and mood/energy signals.  
- **Outputs:** Parsed tasks, mood, profile updates.  
- **Next Nodes:**  
  - Task Manager Agent  
  - Scheduler Agent  
  - Profile Update Agent  
  - Focus Coach Agent

---

### 2. Task Manager Agent (ToDo Agent)
- Stores and updates tasks persistently in Postgres/SQLite.  
- Marks tasks as done/archived when the user mentions them.  
- **Outputs:** Updated task list.

---

### 3. Scheduler Agent
- Places tasks into a calendar view.  
- Color-codes by urgency/priority/difficulty.  
- Detects conflicts between deadlines.  
- **Outputs:** Calendar entries + conflict warnings.

---

### 4. Profile Update Agent
- Updates the user’s profile (name, job, interests, preferences) as learned through conversation.  
- Stored persistently in DB.  
- **Outputs:** Updated profile.

---

### 5. Focus Coach Agent
*(renamed from Mood/Wellness to sound supportive but not clinical)*  
- Interprets the user’s **mood** and **energy**.  
- Suggests appropriate next actions (e.g. “Do an easy task now” vs. “Tackle the hard assignment”).  
- Provides motivational nudges.  
- **Outputs:** Suggested next action + motivation message.

---

### 6. Response Synthesizer
- Gathers outputs from all other agents.  
- Produces a **final natural-language reply** to the user.  
- Explains:  
  - ✅ What tasks/profile were updated  
  - 📅 What was scheduled / conflicts detected  
  - 💡 Suggested next action and motivation  

---

## 🔹 Graph Flow

```mermaid
graph TD
    A[User Message] --> B[Conversation Agent]

    B --> C[Task Manager Agent]
    B --> D[Scheduler Agent]
    B --> E[Profile Update Agent]
    B --> F[Focus Coach Agent]

    C --> G[Response Synthesizer]
    D --> G
    E --> G
    F --> G

    G --> H[END]
