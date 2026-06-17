# build_knowledge.py
# Script that builds the FAISS knowledge base index from stories.txt
# and the SQLite employee database.
# Run this once (or whenever source data changes) to regenerate the index.

import os
import sys

# Force single-threading for BLAS/OpenMP libraries to avoid OpenBLAS memory allocation failure
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Force UTF-8 encoding on standard output to prevent Windows console Unicode crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the project directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import KnowledgeBase
import database

def load_db_knowledge():
    """
    Loads Q&A pairs and facts from the SQLite company database tables:
    company_faq and company_facts.
    """
    qa_pairs = database.get_all_faq()
    facts = database.get_all_facts()
    return qa_pairs, facts


def generate_db_qa_pairs():
    """
    Generates additional Q&A pairs from the SQLite employee database.
    Covers: employee identity, salary, department membership, and
    per-department listings.

    Returns:
        List of (question, answer) tuples.
    """
    qa_pairs = []

    # --- Per-employee Q&A ---
    employees = database.get_all_employees()
    for name, role, department, salary, reports_to in employees:
        reports_to_str = reports_to if reports_to else "None"

        # Who is {name}?
        qa_pairs.append((
            f"Who is {name}?",
            f"{name} is a {role} in the {department} department. "
            f"Salary: {salary} rupees. Reports to: {reports_to_str}."
        ))

        # What is {name}'s salary?
        qa_pairs.append((
            f"What is {name}'s salary?",
            f"{name}'s salary is {salary} rupees per month."
        ))

        # What department does {name} work in?
        qa_pairs.append((
            f"What department does {name} work in?",
            f"{name} works in the {department} department."
        ))

    # --- Per-department Q&A ---
    departments = database.get_departments()
    for dept in departments:
        members = database.get_department_employees(dept)
        names_str = ", ".join(members)
        qa_pairs.append((
            f"Who works in {dept}?",
            f"The following employees work in {dept}: {names_str}."
        ))

    return qa_pairs


def main():
    """Entry point: parse stories, generate DB Q&A, build the index."""
    print("=" * 60)
    print("  BrightTech Solutions — Knowledge Base Builder")
    print("=" * 60)

    # Step 1 & 2 & 3: Load knowledge from SQLite database
    print("\n[1/3] Loading facts and FAQs from SQLite database ...")
    story_qa, facts = load_db_knowledge()
    print(f"      Loaded {len(story_qa)} Q&A pairs from company_faq table")
    print(f"      Loaded {len(facts)} facts from company_facts table")

    # Step 4: Generate DB-derived Q&A pairs
    print("\n[2/3] Generating Q&A pairs from employee database ...")
    db_qa = generate_db_qa_pairs()
    print(f"      Generated {len(db_qa)} Q&A pairs from employee database")

    # Combine all Q&A pairs
    all_qa = story_qa + db_qa
    total_items = len(all_qa) + len(facts)

    # Step 5: Build the knowledge base index
    print("\n[3/3] Building FAISS index ...")
    kb = KnowledgeBase()
    kb.build(all_qa, facts)

    # Step 6: Summary
    print("\n" + "-" * 60)
    print(f"  Total Q&A pairs indexed : {len(all_qa)}")
    print(f"  Total facts indexed     : {len(facts)}")
    print(f"  Total items in index    : {total_items}")
    print("-" * 60)
    print("  Knowledge base built successfully!")
    print("  Index saved to: knowledge.index")
    print("  Metadata saved to: knowledge_data.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
