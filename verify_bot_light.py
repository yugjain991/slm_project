# verify_bot_light.py
# Lightweight verification script for BrightTech Solutions chatbot.
# Test a batch of queries against the database and lightweight fuzzy matcher.

import os
import sys

# Force UTF-8 output to avoid Windows console encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import classify_query, get_small_talk_response
from main_light import (
    handle_employee_lookup,
    handle_salary_query,
    handle_department_query,
    handle_manager_query,
    handle_company_info,
    get_lightweight_kb_answer,
)

test_queries = [
    # Small talk
    "Hi, how are you?",
    "Who are you?",
    
    # Employee Lookup & Roles
    "Who is Priya Verma?",
    "Tell me about Rahul Sharma",
    "Who is the HR Manager?",
    "Who is a Software Engineer?",
    
    # Manager & Reports Queries
    "Who does Amit Kumar report to?",
    "Who reports to Rahul Sharma?",
    "Who is the manager of Priya Verma?",
    "List all managers",
    
    # Salary Statistics
    "What is the average salary in the IT department?",
    "What is the average company salary?",
    "What is the total salary of Sales?",
    "What is the total salary budget of the company?",
    "Who has the highest salary?",
    "What is Amit Kumar's salary?",
    
    # Department Headcount & Details
    "How many employees work in IT?",
    "What is the headcount of the company?",
    "Who works in Sales?",
    "List all departments",
    
    # Company info
    "Where is BrightTech Solutions located?",
    "When was the company founded?",
    
    # Company general knowledge (FAQ / RAG)
    "What services does BrightTech Solutions provide?",
    "What are the office hours?",
    "Does the company provide health insurance?",
    "What are the company values?",
]

print("\n=================== STARTING LIGHTWEIGHT BOT VERIFICATION ===================\n")

for q in test_queries:
    intent = classify_query(q)
    answer = None
    source = None

    if intent == "small_talk":
        answer = get_small_talk_response(q)
        source = "Assistant"
    elif intent == "manager_query":
        answer, source = handle_manager_query(q)
    elif intent == "employee_lookup":
        answer, source = handle_employee_lookup(q)
    elif intent == "salary_query":
        answer, source = handle_salary_query(q)
    elif intent == "department_query":
        answer, source = handle_department_query(q)
    elif intent == "company_info":
        answer, source = handle_company_info(q)

    # Check Lightweight KB if database returned no answer
    if answer is None:
        kb_answer = get_lightweight_kb_answer(q)
        if kb_answer:
            answer = kb_answer
            source = "Lightweight KB (difflib/Jaccard)"

    # Fallback
    if not answer:
        answer = "No database or KB answer. (Falls back to Help suggestions)"
        source = "Fallback"

    print(f"User: {q}")
    print(f"Bot : {answer}")
    print(f"Src : [{source}]\n")

print("=================== LIGHTWEIGHT BOT VERIFICATION COMPLETE ===================")
