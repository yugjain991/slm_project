# router.py
# Smart query router/classifier for BrightTech Solutions chatbot.
# Determines which system layer (small talk, DB lookup, RAG, GPT)
# should handle a given user query.

import re
import difflib
import database


def classify_query(query):
    """
    Classifies a user query into one of the following categories:
        'small_talk'        — greetings, farewell, thanks, etc.
        'manager_query'     — questions about manager/employee reporting relationships
        'employee_lookup'   — asking about a specific person or role
        'salary_query'      — questions about salary, pay, statistics, budgets
        'department_query'  — questions about departments, listings, headcounts
        'company_info'      — questions about CEO, CTO, location, etc.
        'company_knowledge' — general company questions (services, values, mission)
        'general'           — anything else (routed to the GPT model)

    Args:
        query: the user's input string.

    Returns:
        A classification string.
    """
    q = query.lower().strip()

    # ------------------------------------------------------------------
    # 1. Small talk patterns
    # ------------------------------------------------------------------
    small_talk_patterns = [
        r"\b(hi|hello|hey|hola|greetings|hlo|hii|helo|sup)\b",
        r"\bhow are you\b",
        r"\b(thank|thanks|thank you)\b",
        r"\b(bye|goodbye|exit|quit|see you)\b",
        r"\b(who are you|about you|what are you)\b",
        r"\b(good morning|good afternoon|good evening|good night)\b",
    ]
    for pattern in small_talk_patterns:
        if re.search(pattern, q):
            return "small_talk"

    # ------------------------------------------------------------------
    # 2. Manager / Reports patterns
    # ------------------------------------------------------------------
    manager_patterns = [
        r"\breports\s+to\b",
        r"\breport\s+to\b",
        r"\bwho\s+reports\b",
        r"\bmanager\s+of\b",
        r"\bboss\s+of\b",
        r"\bsupervisor\s+of\b",
        r"\bwho\s+is\s+(\w+\s+)?manager\b",
        r"\bwho\s+is\s+(\w+\s+)?boss\b",
        r"\bwho\s+is\s+(\w+\s+)?supervisor\b",
        r"\bwho\s+are\s+the\s+managers\b",
        r"\blist\s+(all\s+)?managers\b",
        r"\bsubordinate(s)?\b",
        r"\bdirect\s+report(s)?\b",
    ]
    for pattern in manager_patterns:
        if re.search(pattern, q):
            return "manager_query"

    # ------------------------------------------------------------------
    # 3. Employee name or Role lookup
    # ------------------------------------------------------------------
    all_employees = database.get_all_employees()
    all_names = [emp[0] for emp in all_employees]

    # Check each known employee name against the query
    for name in all_names:
        # Check full name
        if name.lower() in q:
            return "employee_lookup"
        # Check first name and last name individually (minimum length 3)
        for part in name.lower().split():
            if len(part) > 2 and part in q:
                return "employee_lookup"

    # Check for known employee roles in the query
    all_roles = list(set([emp[1].lower() for emp in all_employees]))
    for role in all_roles:
        if role in q:
            return "employee_lookup"

    # Fuzzy match: split query into words and compare against names
    query_words = q.split()
    for name in all_names:
        name_parts = name.lower().split()
        for name_part in name_parts:
            matches = difflib.get_close_matches(
                name_part, query_words, n=1, cutoff=0.6
            )
            if matches:
                return "employee_lookup"

    # ------------------------------------------------------------------
    # 4. Salary-related keywords
    # ------------------------------------------------------------------
    salary_keywords = [
        r"\bsalar(y|ies)\b",
        r"\bpay\b",
        r"\bearning(s)?\b",
        r"\bcompensation\b",
        r"\bhighest paid\b",
        r"\blowest paid\b",
        r"\bhighest salary\b",
        r"\blowest salary\b",
        r"\bincome\b",
        r"\bwage(s)?\b",
        r"\bctc\b",
        r"\bhow much.*(earn|make|paid)\b",
        r"\bbudget\b",
    ]
    for pattern in salary_keywords:
        if re.search(pattern, q):
            return "salary_query"

    # ------------------------------------------------------------------
    # 5. Department-related keywords
    # ------------------------------------------------------------------
    department_keywords = [
        r"\b(department|departments|team|teams)\b",
        r"\bwho works in\b",
        r"\b(hr|sales|marketing|finance|operations|it department|it team)\b",
        r"\bworks? in\b",
        r"\bwork in\b",
        r"\bheadcount\b",
        r"\bhow many employees\b",
        r"\bnumber of employees\b",
        r"\bstaff count\b",
    ]
    for pattern in department_keywords:
        if re.search(pattern, q):
            return "department_query"

    # ------------------------------------------------------------------
    # 6. Company info keywords (specific facts)
    # ------------------------------------------------------------------
    company_info_keywords = [
        r"\bceo\b",
        r"\bcto\b",
        r"\blocation\b",
        r"\bwhere.*(located|based|office)\b",
        r"\bfounded\b",
        r"\bwhen.*(start|found|establish)\b",
        r"\bcompany name\b",
        r"\bname of.*(company|organization)\b",
        r"\bheadquarter(s)?\b",
    ]
    for pattern in company_info_keywords:
        if re.search(pattern, q):
            return "company_info"

    # ------------------------------------------------------------------
    # 7. General company knowledge (broader topics)
    # ------------------------------------------------------------------
    company_knowledge_keywords = [
        r"\bcompany\b",
        r"\bbright\s?tech\b",
        r"\bservice(s)?\b",
        r"\bproduct(s)?\b",
        r"\bmission\b",
        r"\bvision\b",
        r"\bvalue(s)?\b",
        r"\bculture\b",
        r"\babout\b",
        r"\bwhat do(es)?\b",
        r"\bpolic(y|ies)\b",
        r"\bbenefits?\b",
        r"\bhistory\b",
        r"\bclients?\b",
        r"\bcustomer(s)?\b",
        r"\bproject(s)?\b",
    ]
    for pattern in company_knowledge_keywords:
        if re.search(pattern, q):
            return "company_knowledge"

    # ------------------------------------------------------------------
    # 7. Default: general (routed to GPT/LLM)
    # ------------------------------------------------------------------
    return "general"


def get_small_talk_response(query):
    """
    Returns an appropriate canned response for small-talk queries.

    Args:
        query: the user's input string.

    Returns:
        A friendly response string.
    """
    q = query.lower().strip()

    # Greeting
    if re.search(r"\b(hi|hello|hey|hola|greetings|hlo|hii|helo|sup|good morning|good afternoon|good evening)\b", q):
        return "Hello! I am BrightTech Solutions AI Assistant. How can I help you?"

    # How are you
    if re.search(r"\bhow are you\b", q):
        return "I am doing great! How can I help you today?"

    # Thanks
    if re.search(r"\b(thank|thanks|thank you)\b", q):
        return "You are welcome! Is there anything else I can help with?"

    # Farewell
    if re.search(r"\b(bye|goodbye|exit|quit|see you|good night)\b", q):
        return "Goodbye! Have a great day!"

    # Identity
    if re.search(r"\b(who are you|about you|what are you)\b", q):
        return (
            "I am a smart AI assistant for BrightTech Solutions. "
            "I can answer questions about the company, employees, departments, and more."
        )

    # Default small talk
    return "I am here to help! Try asking about BrightTech Solutions, employees, or departments."
