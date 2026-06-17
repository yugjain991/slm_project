"""
===========================================
  BrightTech Solutions AI Assistant
  Unified Chat Interface (main.py)
===========================================

This is the main entry point for the chatbot.
It routes queries through 3 layers:
  Layer 1: Structured Database (employees, company info)
  Layer 2: RAG Knowledge Base (semantic Q&A search)
  Layer 3: GPT Model (general text generation)
"""

import os
import sys

# Force single-threading for BLAS/OpenMP libraries to avoid OpenBLAS memory allocation failure
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# pyrefly: ignore [missing-import]
import torch
import pickle
import re

# Force UTF-8 encoding on standard output to prevent Windows console Unicode crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==================================
# IMPORTS FROM PROJECT MODULES
# ==================================

from database import (
    get_company_info,
    get_all_company_info,
    get_employee,
    get_all_employees,
    get_department_employees,
    get_departments,
    get_highest_salary,
    get_lowest_salary,
    search_employee,
    format_employee,
    get_manager,
    get_reports_to,
    get_department_stats,
    get_company_stats,
    get_employees_by_role,
    get_all_managers,
)

from router import classify_query, get_small_talk_response
from knowledge_base import KnowledgeBase
from model import GPTLanguageModel


# ==================================
# LOAD KNOWLEDGE BASE (LAYER 2)
# ==================================

def load_knowledge_base():
    """Load the RAG knowledge base."""
    kb = KnowledgeBase()
    if kb.index is not None:
        print("[OK] Knowledge Base loaded.")
        return kb
    else:
        print("[!!] Knowledge Base not found. Run: python build_knowledge.py")
        return None


# ==================================
# LOAD GPT MODEL (LAYER 3)
# ==================================

def load_gpt_model():
    """Load the trained GPT model and vocabulary."""

    vocab_path = "vocab.pkl"
    model_path = "model.pth"

    if not os.path.exists(vocab_path):
        print("[!!] vocab.pkl not found. Run: python train.py")
        return None, None, None

    if not os.path.exists(model_path):
        print("[!!] model.pth not found. Run: python train.py")
        return None, None, None

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    stoi = vocab_data["stoi"]
    itos = vocab_data["itos"]

    vocab_size = len(stoi)

    # Load model
    model = GPTLanguageModel(vocab_size)

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model.eval()
    print(f"[OK] GPT Model loaded ({model.count_parameters():,} parameters).")

    return model, stoi, itos


# ==================================
# GPT GENERATION HELPERS
# ==================================

def encode_text(text, stoi):
    """Encode text to token indices using word-level tokenizer."""
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\s]", text.lower())
    return [stoi.get(t, stoi.get("<UNK>", 1)) for t in tokens]


def decode_tokens(indices, itos):
    """Decode token indices back to text."""
    words = []
    for idx in indices:
        token = itos.get(idx, "<UNK>")
        if token in ("<PAD>", "<END>"):
            break
        if token == "<UNK>":
            continue
        words.append(token)
    return " ".join(words)


def generate_response(prompt, model, stoi, itos, max_tokens=50):
    """Generate a response using the GPT model."""
    tokens = encode_text(prompt, stoi)

    if not tokens:
        return "I couldn't process that input."

    # Limit to block_size
    block_size = model.block_size
    tokens = tokens[-block_size:]

    x = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=max_tokens)

    generated = output[0].tolist()[len(tokens):]
    return decode_tokens(generated, itos)


# ==================================
# LAYER 1: DATABASE HANDLER
# ==================================

def handle_employee_lookup(query):
    """Handle queries about specific employees or roles."""
    emp = search_employee(query)
    if emp:
        reports = get_reports_to(emp[0])
        reports_str = f" | Direct Reports: {', '.join(reports)}" if reports else ""
        return format_employee(emp) + reports_str, "Database"

    all_employees = get_all_employees()
    all_roles = list(set([e[1] for e in all_employees]))
    q = query.lower()
    for role in all_roles:
        if role.lower() in q:
            matching_employees = get_employees_by_role(role)
            if matching_employees:
                names = [e[0] for e in matching_employees]
                return f"The employee(s) with the role of {role} are: {', '.join(names)}.", "Database"

    return None, None


def handle_salary_query(query):
    """Handle salary-related queries, including statistics and individual salaries."""
    q = query.lower()

    depts = get_departments()
    dept_map = {
        "hr": "HR", "human resource": "HR", "human resources": "HR",
        "it": "IT", "information technology": "IT", "tech": "IT",
        "sale": "Sales", "sales": "Sales",
        "marketing": "Marketing",
        "finance": "Finance", "financial": "Finance",
        "operation": "Operations", "operations": "Operations",
        "management": "Management",
    }
    target_dept = None
    for keyword, dept_name in dept_map.items():
        if keyword in q:
            target_dept = dept_name
            break

    if "average" in q or "mean" in q:
        if target_dept:
            stats = get_department_stats(target_dept)
            if stats:
                return (
                    f"The average salary in the {stats['department']} department "
                    f"is ₹{stats['average_salary']:,} per month."
                ), "Database"
        else:
            stats = get_company_stats()
            if stats:
                return (
                    f"The average salary across the company "
                    f"is ₹{stats['average_salary']:,} per month."
                ), "Database"

    if "total" in q or "budget" in q or "sum" in q:
        if target_dept:
            stats = get_department_stats(target_dept)
            if stats:
                return (
                    f"The total salary budget for the {stats['department']} department "
                    f"is ₹{stats['total_salary']:,} per month."
                ), "Database"
        else:
            stats = get_company_stats()
            if stats:
                return (
                    f"The total monthly salary budget for the company "
                    f"is ₹{stats['total_salary']:,}."
                ), "Database"

    if "highest" in q or "most" in q or "top" in q or "maximum" in q:
        result = get_highest_salary()
        if result:
            name, salary = result
            return (
                f"{name} has the highest salary in the company "
                f"at ₹{salary:,} per month."
            ), "Database"

    if "lowest" in q or "least" in q or "minimum" in q:
        result = get_lowest_salary()
        if result:
            name, salary = result
            return (
                f"{name} has the lowest salary in the company "
                f"at ₹{salary:,} per month."
            ), "Database"

    emp = search_employee(query)
    if emp:
        name, role, dept, salary, manager = emp
        return (
            f"{name}'s salary is ₹{salary:,} per month. "
            f"({role} in {dept} department)"
        ), "Database"

    return None, None


def handle_department_query(query):
    """Handle department-related queries, including listings and headcounts."""
    q = query.lower()

    if "all department" in q or "list department" in q or ("departments" in q and ("list" in q or "what" in q)):
        depts = get_departments()
        return (
            f"BrightTech Solutions has the following departments: "
            f"{', '.join(depts)}."
        ), "Database"

    dept_map = {
        "hr": "HR", "human resource": "HR", "human resources": "HR",
        "it": "IT", "information technology": "IT", "tech": "IT",
        "sale": "Sales", "sales": "Sales",
        "marketing": "Marketing",
        "finance": "Finance", "financial": "Finance",
        "operation": "Operations", "operations": "Operations",
        "management": "Management",
    }
    target_dept = None
    for keyword, dept_name in dept_map.items():
        if keyword in q:
            target_dept = dept_name
            break

    if "headcount" in q or "how many" in q or "number of" in q or "staff" in q or "size" in q:
        if target_dept:
            stats = get_department_stats(target_dept)
            if stats:
                return (
                    f"There are {stats['count']} employees working in the "
                    f"{stats['department']} department."
                ), "Database"
        else:
            stats = get_company_stats()
            if stats:
                return (
                    f"BrightTech Solutions has a total headcount of {stats['count']} employees."
                ), "Database"

    if target_dept:
        employees = get_department_employees(target_dept)
        if employees:
            return (
                f"The following employees work in the {target_dept} "
                f"department: {', '.join(employees)}."
            ), "Database"

    return None, None


def handle_manager_query(query):
    """Handle manager and employee reporting relationship queries."""
    q = query.lower()

    if "who are the managers" in q or ("list" in q and "managers" in q) or "who is a manager" in q:
        managers = get_all_managers()
        return f"The managers in the company are: {', '.join(managers)}.", "Database"

    all_employees = get_all_employees()
    all_names = [emp[0] for emp in all_employees]
    
    manager_candidate = None
    for name in all_names:
        if name.lower() in q:
            manager_candidate = name
            break
        for part in name.lower().split():
            if len(part) > 2 and part in q:
                manager_candidate = name
                break

    if "reports to" in q or "report to" in q:
        if "who does" in q or "whom does" in q:
            emp = search_employee(query)
            if emp:
                manager_name = get_manager(emp[0])
                if manager_name:
                    return f"{emp[0]} reports directly to {manager_name}.", "Database"
                else:
                    return f"{emp[0]} does not report to anyone (top-level management).", "Database"
        else:
            if manager_candidate:
                reports = get_reports_to(manager_candidate)
                if reports:
                    return f"The following employees report to {manager_candidate}: {', '.join(reports)}.", "Database"
                else:
                    return f"No employees report to {manager_candidate}.", "Database"

    emp = search_employee(query)
    if emp:
        manager_name = get_manager(emp[0])
        if manager_name:
            return f"The manager of {emp[0]} is {manager_name}.", "Database"
        else:
            return f"{emp[0]} does not report to anyone.", "Database"

    if "manager" in q and manager_candidate:
        reports = get_reports_to(manager_candidate)
        if reports:
            return f"{manager_candidate} is a manager. The employees reporting to them are: {', '.join(reports)}.", "Database"
        else:
            return f"{manager_candidate} does not have any direct reports.", "Database"

    return None, None


def handle_company_info(query):
    """Handle basic company info queries."""
    q = query.lower()

    if "ceo" in q or "chief executive" in q:
        ceo = get_company_info("CEO")
        if ceo:
            return f"The CEO of BrightTech Solutions is {ceo}.", "Database"

    if "cto" in q or "chief technology" in q:
        emp = search_employee("CTO")
        if emp:
            return f"The CTO of BrightTech Solutions is {emp[0]}.", "Database"

    if "location" in q or "where" in q or "office" in q or "located" in q:
        loc = get_company_info("Location")
        if loc:
            return (
                f"BrightTech Solutions is located in {loc}. "
                f"The company has offices in Delhi, Mumbai, and Bangalore."
            ), "Database"

    if "founded" in q or "when" in q or "established" in q or "started" in q:
        founded = get_company_info("Founded")
        if founded:
            return (
                f"BrightTech Solutions was founded in {founded}."
            ), "Database"

    if "company" in q or "brighttech" in q or "bright tech" in q:
        info = get_all_company_info()
        if info:
            return (
                f"{info.get('Company', 'BrightTech Solutions')} is a "
                f"software development company located in "
                f"{info.get('Location', 'India')}. "
                f"CEO: {info.get('CEO', 'N/A')}. "
                f"Founded in {info.get('Founded', 'N/A')}."
            ), "Database"

    return None, None


# ==================================
# MAIN CHAT LOOP
# ==================================

def main():
    """Main chat loop."""

    print()
    print("=" * 50)
    print("   BrightTech Solutions AI Assistant")
    print("=" * 50)
    print()
    print("  I can answer questions about:")
    print("  • Company info (CEO, location, services)")
    print("  • Employees (roles, salaries, departments)")
    print("  • General knowledge about BrightTech")
    print()
    print("  Type 'exit' or 'quit' to end the chat.")
    print("=" * 50)
    print()

    # Load all systems
    kb = load_knowledge_base()
    model, stoi, itos = load_gpt_model()
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Have a great day!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print("Bot: Goodbye! Have a great day!")
            break

        # ---- Step 1: Classify the query ----
        intent = classify_query(user_input)

        answer = None
        source = None

        # ---- Step 2: Handle based on intent ----

        # Small talk
        if intent == "small_talk":
            answer = get_small_talk_response(user_input)
            source = "Assistant"

        # Manager / Reports queries
        elif intent == "manager_query":
            answer, source = handle_manager_query(user_input)

        # Employee lookup
        elif intent == "employee_lookup":
            answer, source = handle_employee_lookup(user_input)

        # Salary query
        elif intent == "salary_query":
            answer, source = handle_salary_query(user_input)

        # Department query
        elif intent == "department_query":
            answer, source = handle_department_query(user_input)

        # Company info
        elif intent == "company_info":
            answer, source = handle_company_info(user_input)

        # ---- Step 3: Try Knowledge Base (RAG) ----
        if answer is None and kb is not None:
            kb_answer = kb.get_best_answer(user_input, threshold=0.5)
            if kb_answer:
                answer = kb_answer
                source = "Knowledge Base"

        # ---- Step 4: Try GPT Model ----
        if answer is None and model is not None:
            answer = generate_response(
                user_input, model, stoi, itos, max_tokens=50
            )
            source = "GPT Model"

        # ---- Step 5: Fallback ----
        if not answer:
            answer = (
                "I'm sorry, I don't have enough information to answer "
                "that question. Try asking about BrightTech Solutions, "
                "employees, departments, or salaries."
            )
            source = "Fallback"

        # ---- Display response ----
        print(f"Bot: {answer}")
        print(f"     [{source}]")
        print()


# ==================================
# ENTRY POINT
# ==================================

if __name__ == "__main__":
    main()
