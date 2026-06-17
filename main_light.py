# main_light.py
# Lightweight, PyTorch-free entry point for the BrightTech Solutions chatbot.
# Routes queries through SQLite database lookups and a lightweight fuzzy search engine.

import os
import sys
import re
import difflib

# Force UTF-8 output to avoid Windows console encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# ==================================
# LAYER 1: DATABASE HANDLERS
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
        "it department": "IT", "it team": "IT", "information technology": "IT", "tech": "IT",
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
        "it department": "IT", "it team": "IT", "information technology": "IT", "tech": "IT",
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
# LIGHTWEIGHT KB (REPLACES RAG/FAISS)
# ==================================

def get_lightweight_kb_answer(query):
    """
    Finds the best matching FAQ or Fact using a robust keyword-overlap metric
    ignoring common stop words, tailored for short user queries against long text.
    """
    import database
    faqs = database.get_all_faq()
    facts = database.get_all_facts()

    stop_words = {"what", "is", "the", "are", "do", "does", "did", "how", "many", "a", "an", "of", "in", "to", "for", "on", "about", "can", "you", "tell", "me", "i", "my", "we", "our"}
    q_words = set([w for w in re.findall(r"\w+", query.lower()) if w not in stop_words])
    
    if not q_words:
        return None

    best_answer = None
    best_score = 0

    # Helper function to calculate score
    def calculate_score(text_to_search):
        target_words = set(re.findall(r"\w+", text_to_search.lower()))
        # Count how many of our important query words appear in the target
        matches = len(q_words.intersection(target_words))
        # Add a slight bonus for exact phrase matching
        phrase_bonus = 0.5 if any(word in text_to_search.lower() for word in q_words) else 0
        return (matches / len(q_words)) + phrase_bonus

    # 1. Search FAQ Questions
    for faq_q, faq_a in faqs:
        score = calculate_score(faq_q)
        if score > best_score:
            best_score = score
            best_answer = faq_a

    # 2. Search Company Facts (compare against the fact text itself)
    for fact in facts:
        score = calculate_score(fact)
        # Give facts a slightly lower priority than direct FAQ matches
        score = score * 0.9 
        if score > best_score:
            best_score = score
            best_answer = fact

    # Threshold: At least half the keywords must match + phrase bonus
    if best_score >= 0.8: 
        return best_answer
    
    return None


# ==================================
# MAIN CHAT LOOP
# ==================================

def get_bot_response(user_input):
    """Core logic to generate a response for a given user input."""
    if not user_input.strip():
        return "", ""

    intent = classify_query(user_input)
    answer = None
    source = None

    if intent == "small_talk":
        answer = get_small_talk_response(user_input)
        source = "Assistant"
    elif intent == "manager_query":
        answer, source = handle_manager_query(user_input)
    elif intent == "employee_lookup":
        answer, source = handle_employee_lookup(user_input)
    elif intent == "salary_query":
        answer, source = handle_salary_query(user_input)
    elif intent == "department_query":
        answer, source = handle_department_query(user_input)
    elif intent == "company_info":
        answer, source = handle_company_info(user_input)

    if answer is None:
        kb_answer = get_lightweight_kb_answer(user_input)
        if kb_answer:
            answer = kb_answer
            source = "Lightweight KB"

    if not answer:
        answer = (
            "I'm sorry, I don't have enough information to answer "
            "that question. Try asking about BrightTech Solutions, "
            "employees, departments, or salaries."
        )
        source = "Fallback"

    return answer, source


def main():
    print()
    print("=" * 60)
    print("   BrightTech Solutions AI Assistant (Lightweight version)")
    print("   [Runs entirely locally without PyTorch/Transformers]")
    print("=" * 60)
    print()
    print("  I can answer questions about:")
    print("  • Company info (CEO, location, services, etc.)")
    print("  • Employees (roles, salaries, managers, departments)")
    print("  • General company policy & values")
    print()
    print("  Type 'exit' or 'quit' to end the chat.")
    print("=" * 60)
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

        # ---- Get Response ----
        answer, source = get_bot_response(user_input)

        # ---- Display response ----
        print(f"Bot: {answer}")
        print(f"     [{source}]")
        print()


if __name__ == "__main__":
    main()
