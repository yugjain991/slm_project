# database.py
# Consolidated database layer for BrightTech Solutions company.db
# Provides CRUD operations for employees and company_info tables.

import sqlite3
import os
import difflib

# Path to the SQLite database file (same directory as this script)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company.db")


def get_connection():
    """Returns a sqlite3 connection to company.db."""
    conn = sqlite3.connect(DB_PATH)
    return conn


def parse_stories_file(filepath):
    """Parses stories.txt into QA pairs and standalone facts."""
    qa_pairs = []
    facts = []

    if not os.path.exists(filepath):
        return qa_pairs, facts

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("Answer:"):
                answer = lines[i + 1].strip()[len("Answer:"):].strip()
                answer = answer.replace("<END>", "").strip()
                qa_pairs.append((question, answer))
                i += 2
                continue
        elif line.startswith("Answer:"):
            i += 1
            continue
        elif line:
            clean_line = line.replace("<END>", "").strip()
            if clean_line:
                facts.append(clean_line)
        i += 1

    return qa_pairs, facts


def init_db():
    """
    Initializes the database with schema and seed data.
    Creates employees, company_info, company_facts, and company_faq tables,
    then populates them.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # --- Create tables ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                name TEXT,
                role TEXT,
                department TEXT,
                salary INTEGER,
                reports_to TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_facts (
                fact TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_faq (
                question TEXT,
                answer TEXT
            )
        """)

        # --- Seed employee data (clear first to avoid duplicates on re-init) ---
        cursor.execute("DELETE FROM employees")
        employees = [
            ("Rahul Sharma", "CEO", "Management", 200000, None),
            ("Priya Verma", "CTO", "IT", 180000, "Rahul Sharma"),
            ("Neha Gupta", "HR Manager", "HR", 120000, "Rahul Sharma"),
            ("Amit Kumar", "Software Engineer", "IT", 70000, "Priya Verma"),
            ("Rohit Singh", "Sales Executive", "Sales", 60000, "Rahul Sharma"),
            ("Ananya Jain", "Marketing Manager", "Marketing", 75000, "Rahul Sharma"),
            ("Karan Mehta", "Financial Analyst", "Finance", 80000, "Rahul Sharma"),
            ("Sneha Patel", "Operations Manager", "Operations", 85000, "Rahul Sharma"),
            ("Anshuman Verma", "Operations Specialist", "Operations", 85000, "Rahul Sharma"),
        ]
        cursor.executemany(
            "INSERT INTO employees (name, role, department, salary, reports_to) VALUES (?, ?, ?, ?, ?)",
            employees
        )

        # --- Seed company info ---
        cursor.execute("DELETE FROM company_info")
        company_info = [
            ("CEO", "Rahul Sharma"),
            ("Company", "BrightTech Solutions"),
            ("Location", "Bangalore, India"),
            ("Founded", "2015"),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO company_info (key, value) VALUES (?, ?)",
            company_info
        )

        # --- Seed company facts & FAQ from stories.txt ---
        cursor.execute("DELETE FROM company_facts")
        cursor.execute("DELETE FROM company_faq")

        stories_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stories.txt")
        if os.path.exists(stories_path):
            qa_pairs, facts = parse_stories_file(stories_path)
            cursor.executemany("INSERT INTO company_faq (question, answer) VALUES (?, ?)", qa_pairs)
            cursor.executemany("INSERT INTO company_facts (fact) VALUES (?)", [(f,) for f in facts])

        conn.commit()


def get_company_info(key):
    """
    Looks up a single value from the company_info table.
    Case-insensitive key matching.
    Returns the value string or None if not found.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM company_info WHERE LOWER(key) = LOWER(?)", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None


def get_all_company_info():
    """
    Returns a dict of all company info key-value pairs.
    Example: {'CEO': 'Rahul Sharma', 'Company': 'BrightTech Solutions', ...}
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM company_info")
        return {row[0]: row[1] for row in cursor.fetchall()}


def get_employee(name):
    """
    Finds an employee by exact or partial name match (using SQL LIKE).
    Returns a tuple (name, role, department, salary, reports_to) or None.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, role, department, salary, reports_to FROM employees WHERE name LIKE ?",
            (f"%{name}%",)
        )
        row = cursor.fetchone()
        return row if row else None


def get_all_employees():
    """Returns a list of all employee tuples (name, role, department, salary, reports_to)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, role, department, salary, reports_to FROM employees")
        return cursor.fetchall()


def get_department_employees(dept):
    """
    Returns a list of employee names belonging to the given department.
    Case-insensitive department matching.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM employees WHERE LOWER(department) = LOWER(?)", (dept,)
        )
        return [row[0] for row in cursor.fetchall()]


def get_departments():
    """Returns a list of distinct department names."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT department FROM employees")
        return [row[0] for row in cursor.fetchall()]


def get_highest_salary():
    """Returns (name, salary) tuple for the highest-paid employee."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"
        )
        return cursor.fetchone()


def get_lowest_salary():
    """Returns (name, salary) tuple for the lowest-paid employee."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, salary FROM employees ORDER BY salary ASC LIMIT 1"
        )
        return cursor.fetchone()


def search_employee(query):
    """
    Uses difflib.get_close_matches for fuzzy name matching.
    Tries exact/partial match first via get_employee(), then falls back
    to fuzzy matching against all employee names.
    Returns an employee tuple (name, role, department, salary, reports_to) or None.
    """
    # Try exact/partial match first
    result = get_employee(query)
    if result:
        return result

    # Fall back to fuzzy matching
    all_employees = get_all_employees()
    all_names = [emp[0] for emp in all_employees]

    matches = difflib.get_close_matches(query, all_names, n=1, cutoff=0.4)
    if matches:
        # Retrieve the full employee record for the best fuzzy match
        return get_employee(matches[0])

    return None


def format_employee(emp):
    """
    Takes an employee tuple (name, role, department, salary, reports_to)
    and returns a nicely formatted string.
    """
    name, role, department, salary, reports_to = emp
    reports_to_str = reports_to if reports_to else "None"
    return (
        f"Name: {name} | Role: {role} | Department: {department} "
        f"| Salary: \u20b9{salary} | Reports to: {reports_to_str}"
    )


def add_employee(name, role, department, salary, reports_to=None):
    """Inserts a new employee record or replaces an existing one."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO employees (name, role, department, salary, reports_to) VALUES (?, ?, ?, ?, ?)",
            (name, role, department, salary, reports_to)
        )
        conn.commit()


def get_manager(name):
    """Gets the manager of an employee."""
    emp = search_employee(name)
    if emp:
        return emp[4]  # reports_to
    return None


def get_reports_to(manager_name):
    """Gets all employees reporting directly to the given manager."""
    m_emp = search_employee(manager_name)
    if not m_emp:
        return []
    full_manager_name = m_emp[0]
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM employees WHERE reports_to = ?", (full_manager_name,)
        )
        return [row[0] for row in cursor.fetchall()]


def get_department_stats(dept):
    """Gets headcount, total salary, and average salary for a department."""
    all_depts = get_departments()
    matches = difflib.get_close_matches(dept, all_depts, n=1, cutoff=0.5)
    if not matches:
        for d in all_depts:
            if d.lower() == dept.lower():
                matches = [d]
                break
    if not matches:
        return None
    actual_dept = matches[0]
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), SUM(salary), AVG(salary) FROM employees WHERE department = ?", (actual_dept,)
        )
        row = cursor.fetchone()
        if row and row[0] > 0:
            return {
                "department": actual_dept,
                "count": row[0],
                "total_salary": row[1],
                "average_salary": round(row[2], 2)
            }
    return None


def get_company_stats():
    """Gets headcount, total salary, and average salary of the entire company."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), SUM(salary), AVG(salary) FROM employees"
        )
        row = cursor.fetchone()
        if row and row[0] > 0:
            return {
                "count": row[0],
                "total_salary": row[1],
                "average_salary": round(row[2], 2)
            }
    return None


def get_employees_by_role(role):
    """Gets all employees with a specific role."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, department FROM employees WHERE LOWER(role) LIKE ?", (f"%{role.lower()}%",)
        )
        return cursor.fetchall()


def get_all_managers():
    """Gets all unique manager names from employees table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT reports_to FROM employees WHERE reports_to IS NOT NULL AND reports_to != ''"
        )
        return [row[0] for row in cursor.fetchall()]


def get_all_facts():
    """Gets all facts from the company_facts table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT fact FROM company_facts")
        return [row[0] for row in cursor.fetchall()]


def get_all_faq():
    """Gets all QA pairs from the company_faq table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM company_faq")
        return [(row[0], row[1]) for row in cursor.fetchall()]


# Initialize the database when this module is first imported
init_db()