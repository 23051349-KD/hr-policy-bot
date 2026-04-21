"""
agent.py — HR Policy Bot | Shared Agent Module
Agentic AI Hands-On Course | Capstone Project
Domain: HR Policy Bot
User: Company employees asking about leave, payroll, policies, benefits
Author: [Your Name] | Roll: [Your Roll No] | Batch: Agentic AI 2026

Import this from capstone_streamlit.py — do not run directly.
"""

import os
import re
from datetime import datetime
from typing import TypedDict, List

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — 12 HR Policy Documents
# ─────────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Annual Leave Policy",
        "text": """Annual Leave Policy — Employees are entitled to 18 days of paid annual leave per calendar year.
Leave accrues at 1.5 days per month starting from the date of joining. Employees must serve a probation period
of 3 months before availing annual leave. Leave must be applied at least 7 days in advance through the HRMS
portal for planned leaves. Emergency leaves must be notified to the immediate manager via email or phone within
2 hours of absence. Unused annual leave up to a maximum of 10 days can be carried forward to the next calendar year.
Any leave exceeding the accrued balance will be treated as leave without pay (LWP). Employees must not take more
than 10 consecutive days of annual leave without prior written approval from the department head.
Leave encashment is permitted only at the time of resignation, retirement, or during the annual leave encashment
window in December. Employees on probation are not eligible for leave encashment."""
    },
    {
        "id": "doc_002",
        "topic": "Sick Leave Policy",
        "text": """Sick Leave Policy — All full-time employees are entitled to 10 days of paid sick leave per calendar year.
Sick leave does not carry forward to the following year and cannot be encashed under any circumstances. Employees
must inform their manager before 10:00 AM on the day of absence or as early as possible. For sick leave exceeding
3 consecutive days, a medical certificate from a registered medical practitioner is mandatory. Sick leave is
non-transferable and cannot be combined with other leave types to extend a holiday period. Employees on probation
are entitled to 5 days of sick leave during their probation period. Sick leave taken just before or after a
public holiday will require a medical certificate regardless of duration. Frequent use of sick leave (more than
6 days in a quarter) may result in a formal discussion with HR. In cases of prolonged illness exceeding 30 days,
employees may apply for medical leave under a separate policy subject to management approval."""
    },
    {
        "id": "doc_003",
        "topic": "Work From Home Policy",
        "text": """Work From Home (WFH) Policy — Employees are eligible for up to 2 days of work from home per week,
subject to manager approval and business requirements. WFH requests must be submitted through the HRMS portal
by Friday of the preceding week. Ad hoc WFH requests may be approved by the manager via email or chat.
Employees working from home are expected to be available during core working hours: 10:00 AM to 5:00 PM.
All standard office work norms apply while working from home, including data security policies and
confidentiality requirements. Employees in client-facing roles or those with specific operational requirements
may have different WFH entitlements as defined by their department policy. WFH is not applicable during the
first 3 months of employment or probation. New hires must work from office for the full probation period.
Employees are responsible for ensuring a stable internet connection and a professional work environment while
on WFH days. WFH approval can be revoked if an employee's performance or availability is found to be impacted."""
    },
    {
        "id": "doc_004",
        "topic": "Payroll and Salary Structure",
        "text": """Payroll and Salary Structure — Salaries are processed on the last working day of every month.
The salary structure consists of: Basic Salary (40% of CTC), House Rent Allowance or HRA (20% of Basic),
Special Allowance (variable component), Provident Fund (PF) employer contribution (12% of Basic),
and Professional Tax as per state slab. Employees can view their salary slips on the HRMS portal by the 2nd
of each month. Salary revisions are carried out annually during the April appraisal cycle. A minimum of 6 months
in the current role is required to be eligible for an increment. Off-cycle salary corrections due to payroll
errors must be raised through the HR helpdesk within 7 days of the salary credit. Tax declarations must be
submitted through the HRMS portal by the 15th of April every year for investment proof purposes. Employees
who do not submit proofs by the deadline will have TDS deducted at the maximum applicable slab rate.
Salary is credited directly to the bank account registered in the HRMS system."""
    },
    {
        "id": "doc_005",
        "topic": "Performance Appraisal and Increments",
        "text": """Performance Appraisal Process — The company follows an annual performance appraisal cycle conducted
in March–April. The appraisal process has three stages: self-assessment by the employee, manager review and
rating, and calibration by the HR team. Ratings are on a 5-point scale: 1 (Below Expectations), 2 (Partially
Meets Expectations), 3 (Meets Expectations), 4 (Exceeds Expectations), 5 (Outstanding). Increment percentages
are linked to performance ratings and are announced in April with effect from April 1st. Employees rated 3 and
above are eligible for a performance bonus in addition to the increment. Employees who join between October and
December receive a prorated increment in their first appraisal cycle. Employees on a Performance Improvement
Plan (PIP) are not eligible for an increment until they exit the PIP successfully. All employees must complete
their self-assessment on the HRMS portal before March 31st. Mid-year feedback sessions are conducted in
September to ensure alignment between employee goals and business objectives."""
    },
    {
        "id": "doc_006",
        "topic": "Provident Fund and Gratuity",
        "text": """Provident Fund and Gratuity — The company contributes 12% of the employee's basic salary towards
the Employee Provident Fund (EPF) as mandated by the EPF Act. Employees also contribute 12% of their basic
salary towards EPF. The PF account is registered under the EPFO (Employees' Provident Fund Organisation) and
can be tracked on the EPFO member portal using the UAN number. Employees can withdraw their PF balance under
specific circumstances: resignation and no new employment for 2 months, retirement, or specific emergencies
like medical treatment, house construction, or education as permitted under EPFO rules. Partial withdrawal
is allowed after 5 years of service for housing or medical purposes. Gratuity is payable to employees who
have completed a minimum of 5 continuous years of service with the company. Gratuity is calculated as:
(Last Drawn Basic Salary × 15 × Years of Service) divided by 26. Gratuity is paid at the time of
resignation, retirement, or death/disability. The maximum gratuity payable is INR 20 lakhs as per the
Payment of Gratuity Act."""
    },
    {
        "id": "doc_007",
        "topic": "Code of Conduct and Disciplinary Policy",
        "text": """Code of Conduct and Disciplinary Policy — All employees are expected to maintain the highest
standards of professional behavior, integrity, and respect in the workplace. Any form of harassment, bullying,
discrimination, or misconduct will not be tolerated. Employees must not engage in activities that conflict with
the company's business interests. Use of company resources for personal benefit or unauthorized purposes is
strictly prohibited. Confidential information must not be shared with external parties without explicit
authorization. Disciplinary action is taken progressively: Verbal Warning for minor first-time offences,
Written Warning for repeated or moderate offences, Performance Improvement Plan for sustained performance issues,
Suspension without pay for serious misconduct, and Termination for gross misconduct or repeated policy violations.
All disciplinary proceedings are conducted in accordance with natural justice principles — the employee is
given an opportunity to present their case. Employees can appeal disciplinary decisions to the HR Director
within 10 working days of receiving the decision. The company maintains the right to terminate employment
immediately in cases of fraud, violence, or breach of confidentiality."""
    },
    {
        "id": "doc_008",
        "topic": "Maternity and Paternity Leave",
        "text": """Maternity and Paternity Leave Policy — Female employees are entitled to 26 weeks of paid maternity
leave for the first two children as per the Maternity Benefit (Amendment) Act, 2017. For the third child onwards,
maternity leave is 12 weeks. Employees must submit a medical certificate indicating the expected date of
delivery at least 8 weeks before the leave commences. Maternity leave can start 8 weeks before the expected
delivery date. The remaining leave is available post-delivery. In case of miscarriage or medical termination,
6 weeks of leave is provided with a valid medical certificate. The company also provides a creche facility or
a monthly creche allowance for children under 6 years of age for mothers returning from maternity leave.
Male employees are entitled to 5 days of paid paternity leave, which must be availed within 3 months of the
child's birth or adoption. Paternity leave cannot be split and must be taken in one continuous stretch.
Adoption leave of 12 weeks is available for employees legally adopting a child below 3 months of age."""
    },
    {
        "id": "doc_009",
        "topic": "Employee Benefits and Insurance",
        "text": """Employee Benefits and Insurance — The company provides a comprehensive benefits package to all
permanent employees. Group Health Insurance covers the employee, spouse, and up to 2 dependent children with
a sum insured of INR 3 lakhs per family per year. Top-up insurance is available at an additional premium
deducted from salary. Group Term Life Insurance provides coverage equal to 3x the employee's annual CTC with
no premium deduction from the employee. Personal Accidental Insurance covers permanent disability or death due
to accidents up to 5x the annual CTC. Employees are eligible for an interest-free laptop advance of up to
INR 50,000, repayable in 12 EMIs. A vehicle advance of up to INR 1 lakh is available at 4% per annum after
2 years of service. The company sponsors professional certifications relevant to the employee's role up to
INR 15,000 per financial year subject to manager approval. Gym membership reimbursement of up to INR 5,000
per year is provided under the wellness benefit program. All benefits are subject to employment status and
are discontinued upon resignation or termination."""
    },
    {
        "id": "doc_010",
        "topic": "Resignation and Exit Process",
        "text": """Resignation and Exit Process — Employees wishing to resign must submit a formal resignation letter
or email to their reporting manager with a copy to HR. The notice period for employees in their first year
is 30 days. After the first year, the notice period is 60 days. Senior Manager and above roles have a 90-day
notice period. Notice period buyout is possible with mutual consent at the rate of one day's basic salary per
day of notice not served. Full and Final (F&F) settlement is processed within 45 working days from the last
working day. F&F includes: pending salary, leave encashment (as applicable), and deduction of any outstanding
advances or dues. A detailed exit interview is conducted by HR to capture feedback. The employee must return
all company assets — laptop, access cards, and documents — before the last working day. EPFO transfer or
withdrawal can be initiated after F&F is completed. Service certificates, experience letters, and relieving
letters are issued within 7 working days of the last day."""
    },
    {
        "id": "doc_011",
        "topic": "Anti-Sexual Harassment Policy (POSH)",
        "text": """Prevention of Sexual Harassment (POSH) Policy — The company is committed to providing a safe,
respectful, and inclusive work environment. All forms of sexual harassment — verbal, non-verbal, physical, or
digital — are strictly prohibited under the Sexual Harassment of Women at Workplace (Prevention, Prohibition
and Redressal) Act, 2013. The Internal Committee (IC) is the designated body responsible for receiving and
investigating sexual harassment complaints. Complaints must be submitted in writing to the IC within 3 months
of the incident. The IC will conduct a fair and time-bound inquiry within 90 days of receiving the complaint.
Both the complainant and the respondent are given equal opportunity to present their case. During the inquiry,
the complainant may request interim relief such as transfer or reassignment. The IC's findings are reported to
the management with recommended action. Action can range from a written apology to termination depending on
the severity. Confidentiality is maintained throughout the process. Retaliation against the complainant is
itself a serious offence under this policy. The IC chairman and members' contact details are available on the
company intranet and HRMS portal."""
    },
    {
        "id": "doc_012",
        "topic": "HR Helpdesk and Escalation",
        "text": """HR Helpdesk and Escalation Process — The HR helpdesk is the primary point of contact for all
employee queries related to payroll, leave, policies, benefits, and documentation. Employees can reach the
HR helpdesk via: Email at hr-helpdesk@company.com, Phone at 1800-HR-HELP (1800-47-4357) available Monday to
Friday 9:00 AM to 6:00 PM, or through the self-service HR portal at hr.company.com. Standard response time
for helpdesk queries is 2 working days. Escalation to a Senior HR Business Partner is available if the issue
is not resolved within 5 working days. Critical issues such as non-payment of salary, POSH complaints, or
medical emergencies related to Group Insurance should be marked as URGENT and addressed within 4 hours.
For IT-HR system issues like HRMS portal access or payslip download problems, tickets can be raised on the
IT service desk separately. Employees outside of business hours in cases of genuine emergencies can contact
the HR Emergency Line at +91-9900-HR-SOS (9900-47-767). All HR interactions are treated as confidential."""
    },
]

# ─────────────────────────────────────────────────────────────
# INITIALISATION — Embedder, ChromaDB, LLM
# ─────────────────────────────────────────────────────────────

print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedder loaded")

print("🔄 Building ChromaDB knowledge base...")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="hr_policy_kb",
    metadata={"hnsw:space": "cosine"}
)

doc_texts = [d["text"] for d in DOCUMENTS]
doc_ids   = [d["id"]   for d in DOCUMENTS]
doc_metas = [{"topic": d["topic"]} for d in DOCUMENTS]
doc_embeddings = embedder.encode(doc_texts).tolist()

collection.add(
    documents=doc_texts,
    embeddings=doc_embeddings,
    ids=doc_ids,
    metadatas=doc_metas
)
print(f"✅ ChromaDB loaded with {collection.count()} documents")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("✅ LLM (Groq llama-3.3-70b-versatile) initialised")


# ─────────────────────────────────────────────────────────────
# PART 2 — STATE DESIGN
# ─────────────────────────────────────────────────────────────

class HRBotState(TypedDict):
    # Input
    question:      str
    # Memory
    messages:      List[dict]
    user_name:     str          # HR-specific: remember employee name
    # Routing
    route:         str          # "retrieve" | "memory_only" | "tool"
    # RAG
    retrieved:     str
    sources:       List[str]
    # Tool
    tool_result:   str
    # Output
    answer:        str
    # Evaluation
    faithfulness:  float
    eval_retries:  int


# ─────────────────────────────────────────────────────────────
# PART 3 — NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2


def memory_node(state: HRBotState) -> dict:
    """Appends question to history, applies sliding window, extracts employee name."""
    msgs      = state.get("messages", [])
    user_name = state.get("user_name", "")

    # Extract name if employee introduces themselves
    q = state["question"].lower()
    match = re.search(r"my name is ([a-z ]+)", q)
    if match:
        user_name = match.group(1).strip().title()

    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:           # sliding window — keep last 3 turns
        msgs = msgs[-6:]

    return {"messages": msgs, "user_name": user_name}


def router_node(state: HRBotState) -> dict:
    """Decides which path to take: retrieve, memory_only, or tool."""
    question  = state["question"]
    messages  = state.get("messages", [])
    recent    = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for an HR Policy Bot that helps company employees with HR-related queries.

Available routes:
- retrieve: Search the HR policy knowledge base for leave, payroll, benefits, PF, gratuity,
  performance, resignation, POSH, WFH, or any HR policy question.
- memory_only: Answer from conversation history only (e.g. 'what did you just say?',
  'repeat that', 'my name is ...', 'what was my first question?').
- tool: Use the datetime tool (e.g. 'what is today's date?', 'how many days until month end?',
  'what day is it?').

Recent conversation: {recent}
Current question: {question}

Reply with EXACTLY ONE word: retrieve, memory_only, or tool"""

    result = llm.invoke([HumanMessage(content=prompt)])
    route  = result.content.strip().lower().split()[0]
    if route not in ("retrieve", "memory_only", "tool"):
        route = "retrieve"

    return {"route": route}


def retrieval_node(state: HRBotState) -> dict:
    """Queries ChromaDB for top 3 relevant HR policy chunks."""
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: HRBotState) -> dict:
    """Used for memory-only route — no KB query needed."""
    return {"retrieved": "", "sources": []}


def tool_node(state: HRBotState) -> dict:
    """Datetime tool — returns current date, time, and useful HR date info."""
    try:
        now         = datetime.now()
        day_name    = now.strftime("%A")
        date_str    = now.strftime("%d %B %Y")
        time_str    = now.strftime("%I:%M %p")
        month_name  = now.strftime("%B %Y")
        # Days left in the month (useful for leave/payroll queries)
        import calendar
        days_in_month = calendar.monthrange(now.year, now.month)[1]
        days_left     = days_in_month - now.day

        tool_result = (
            f"Current date: {day_name}, {date_str}\n"
            f"Current time: {time_str}\n"
            f"Current month: {month_name}\n"
            f"Days remaining in this month: {days_left}\n"
            f"Financial year: April {now.year if now.month >= 4 else now.year - 1} "
            f"to March {now.year + 1 if now.month >= 4 else now.year}"
        )
    except Exception as e:
        tool_result = f"Could not fetch date/time information: {str(e)}"

    return {"tool_result": tool_result, "retrieved": "", "sources": []}


def answer_node(state: HRBotState) -> dict:
    """Generates the final answer using KB context, tool output, and chat history."""
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    user_name    = state.get("user_name", "")

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    name_line = f" You are speaking with {user_name}." if user_name else ""

    retry_instruction = (
        "\n\nIMPORTANT: Your previous answer was flagged for potential hallucination. "
        "Be even more conservative — only state what is explicitly in the provided context."
        if eval_retries >= 1 else ""
    )

    system_prompt = f"""You are an intelligent HR Policy Assistant for a corporate organisation.{name_line}
Your role is to help employees understand company HR policies accurately and professionally.

GROUNDING RULE: Answer ONLY from the provided KNOWLEDGE BASE or TOOL RESULT context.
If the answer is not in the context, say: "I don't have specific information about that in our HR policy database.
Please contact the HR helpdesk at hr-helpdesk@company.com or call 1800-HR-HELP (1800-47-4357)."

Do NOT invent policies, numbers, or rules not present in the context.
Be warm, professional, and concise. Use bullet points for multi-step answers.
Always cite which policy area your answer comes from.{retry_instruction}"""

    # Build conversation history for context
    history_msgs = [SystemMessage(content=system_prompt)]
    for msg in messages[:-1]:       # all except the current question
        if msg["role"] == "user":
            history_msgs.append(HumanMessage(content=msg["content"]))
        else:
            history_msgs.append(AIMessage(content=msg["content"]))

    # Add the current question with context
    user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    history_msgs.append(HumanMessage(content=user_content))

    result = llm.invoke(history_msgs)
    return {"answer": result.content, "tool_result": ""}


def eval_node(state: HRBotState) -> dict:
    """Rates faithfulness of the answer against retrieved context."""
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:600]
    retries  = state.get("eval_retries", 0)

    if not context:
        # No retrieval — skip faithfulness check
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful (every claim is in the context).
0.5 = some hallucination.
0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:400]}

Score (0.0 to 1.0):"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        score  = float(re.search(r"[0-9]+\.?[0-9]*", result.content).group())
        score  = max(0.0, min(1.0, score))
    except Exception:
        score  = 0.8    # default pass on parse failure

    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: HRBotState) -> dict:
    """Appends the assistant's answer to conversation history."""
    msgs   = state.get("messages", [])
    answer = state.get("answer", "")
    msgs   = msgs + [{"role": "assistant", "content": answer}]
    return {"messages": msgs}


# ─────────────────────────────────────────────────────────────
# PART 4 — GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────

def route_decision(state: HRBotState) -> str:
    """After router_node: decide which retrieval path to take."""
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: HRBotState) -> str:
    """After eval_node: retry answer or save and finish."""
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"     # retry


graph = StateGraph(HRBotState)

graph.add_node("memory",   memory_node)
graph.add_node("router",   router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip",     skip_retrieval_node)
graph.add_node("tool",     tool_node)
graph.add_node("answer",   answer_node)
graph.add_node("eval",     eval_node)
graph.add_node("save",     save_node)

graph.set_entry_point("memory")

graph.add_edge("memory",   "router")
graph.add_conditional_edges("router", route_decision,
                             {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
graph.add_edge("retrieve", "answer")
graph.add_edge("skip",     "answer")
graph.add_edge("tool",     "answer")
graph.add_edge("answer",   "eval")
graph.add_conditional_edges("eval", eval_decision,
                             {"answer": "answer", "save": "save"})
graph.add_edge("save",     END)

app = graph.compile(checkpointer=MemorySaver())
print("✅ Graph compiled successfully")


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def ask(question: str, thread_id: str = "default") -> dict:
    """Run the HR Policy agent and return the result state."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result
