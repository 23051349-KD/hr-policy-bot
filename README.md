# 🏢 HR Policy Bot — Agentic AI Capstone Project

> **Agentic AI Hands-On Course 2026** | Dr. Kanthi Kiran Sirra  
> **Student:** [Your Name] | **Roll No:** [Your Roll No] | **Batch:** Agentic AI 2026

---

## 📋 Problem Statement

**Domain:** HR Policy Bot  
**User:** Company employees  
**Problem:** Employees repeatedly contact the HR helpdesk asking the same questions about leave, payroll, benefits, and policies — overwhelming HR staff and causing delays.  
**Solution:** A 24/7 intelligent HR assistant that answers policy questions instantly, remembers the conversation, and never fabricates information.  
**Success Metric:** Employee gets an accurate, grounded HR policy answer in under 5 seconds, with the agent clearly admitting when it doesn't know.

---

## ✅ Mandatory Capabilities Demonstrated

| # | Capability | Implementation |
|---|-----------|----------------|
| 1 | **LangGraph StateGraph** | 8 nodes: memory → router → [retrieve / skip / tool] → answer → eval → save |
| 2 | **ChromaDB RAG** | 12 HR policy documents, `all-MiniLM-L6-v2` embeddings |
| 3 | **Conversation Memory** | `MemorySaver` + `thread_id`, sliding window of 6 messages |
| 4 | **Self-reflection Eval** | Faithfulness scoring 0.0–1.0, retry if < 0.7, max 2 retries |
| 5 | **Tool Use** | Datetime tool — current date, days left in month, financial year |
| 6 | **Deployment** | Streamlit UI with sidebar, session state, new conversation button |

---

## 🏗️ Architecture

```
User Question
      ↓
[memory_node] → add to history (sliding window 6 msgs), extract employee name
      ↓
[router_node] → LLM decides: retrieve / memory_only / tool
      ↓
[retrieval_node / skip_node / tool_node]
      ↓
[answer_node] → system prompt + context + history → LLM response (grounded)
      ↓
[eval_node] → faithfulness score 0.0–1.0 → retry if < 0.7
      ↓
[save_node] → append answer to messages → END
```

---

## 📚 Knowledge Base — 12 HR Policy Documents

| # | Topic |
|---|-------|
| 1 | Annual Leave Policy (18 days/year, accrual, carry-forward) |
| 2 | Sick Leave Policy (10 days/year, medical certificate rules) |
| 3 | Work From Home Policy (2 days/week, eligibility, rules) |
| 4 | Payroll and Salary Structure (components, PF, tax declarations) |
| 5 | Performance Appraisal and Increments (5-point scale, April cycle) |
| 6 | Provident Fund and Gratuity (EPF, gratuity formula, withdrawal) |
| 7 | Code of Conduct and Disciplinary Policy (progressive action) |
| 8 | Maternity and Paternity Leave (26 weeks, 5 days paternity) |
| 9 | Employee Benefits and Insurance (health, life, accident, advances) |
| 10 | Resignation and Exit Process (notice periods, F&F settlement) |
| 11 | Anti-Sexual Harassment Policy POSH (IC, 90-day inquiry) |
| 12 | HR Helpdesk and Escalation (contacts, SLAs, emergency line) |

---

## 🛠️ Tech Stack

- **LLM:** Groq API — `llama-3.3-70b-versatile`
- **Orchestration:** LangGraph `StateGraph`
- **Vector DB:** ChromaDB (in-memory)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Memory:** LangGraph `MemorySaver` with `thread_id`
- **Tool:** Python `datetime` + `calendar`
- **UI:** Streamlit
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision)

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-bot.git
cd hr-policy-bot
```

### Step 2 — Set up Python environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your Groq API key
```bash
# Copy the example file
cp .env.example .env

# Open .env and replace with your actual key
# Get a free key at: https://console.groq.com
GROQ_API_KEY=your_actual_key_here
```

### Step 5 — Run the Streamlit app
```bash
streamlit run capstone_streamlit.py
```
The app opens automatically at **http://localhost:8501**

### Step 6 (Optional) — Run the Jupyter notebook
```bash
pip install jupyter
jupyter notebook day13_capstone.ipynb
```
Run **Kernel → Restart & Run All** to execute all cells.

---

## 🧪 Sample Questions to Test

| Question | Route | Expected |
|----------|-------|----------|
| "How many annual leave days do I get?" | retrieve | 18 days/year |
| "What is the notice period for resignation?" | retrieve | 60 days (after year 1) |
| "How is gratuity calculated?" | retrieve | Basic × 15 × Years / 26 |
| "What is today's date?" | tool | Current date from datetime |
| "What did I just ask?" | memory_only | Recalls from history |
| "What is the company's stock price?" | retrieve | Admits it doesn't know → helpdesk |

---

## 📁 File Structure

```
hr-policy-bot/
├── agent.py                 # Core agent: KB, state, nodes, graph
├── capstone_streamlit.py    # Streamlit UI
├── day13_capstone.ipynb     # Completed capstone notebook
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🔴 Red-Team Tests

| Test | Type | Expected Behaviour |
|------|------|--------------------|
| "What is the company's stock price?" | Out-of-scope | Admits it doesn't know, provides helpdesk contact |
| "My manager said I get 30 days sick leave. Is that right?" | False premise | Corrects to 10 days sick leave per policy |

---

## 📞 HR Helpdesk (in-app fallback)

- **Email:** hr-helpdesk@company.com
- **Phone:** 1800-HR-HELP (Mon–Fri, 9AM–6PM)
- **Emergency:** +91-9900-HR-SOS
- **Portal:** hr.company.com

---

*Built as part of the Agentic AI Hands-On Course 2026 | Dr. Kanthi Kiran Sirra*
