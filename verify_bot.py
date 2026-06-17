import os
import sys

# Force single-threading for BLAS/OpenMP libraries to avoid OpenBLAS memory allocation failure
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Force UTF-8 output to avoid Windows console encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from router import classify_query, get_small_talk_response
from knowledge_base import KnowledgeBase
from main import (
    load_knowledge_base,
    handle_employee_lookup,
    handle_salary_query,
    handle_department_query,
    handle_manager_query,
    handle_company_info,
)

# Initialize KB
kb = load_knowledge_base()

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

print("\n=================== STARTING BOT VERIFICATION ===================\n")

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

    # Check KB
    if answer is None and kb is not None:
        kb_answer = kb.get_best_answer(q, threshold=0.5)
        if kb_answer:
            answer = kb_answer
            source = "Knowledge Base"

    # Fallback
    if not answer:
        answer = "No direct database or KB answer. (Falls back to GPT Model)"
        source = "Fallback / GPT"

    print(f"User: {q}")
    print(f"Bot : {answer}")
    print(f"Src : [{source}]\n")

print("=================== BOT VERIFICATION COMPLETE ===================")
