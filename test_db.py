import database

print("=== Running Database Functions Tests ===")

print("\n1. Testing get_highest_salary:")
highest = database.get_highest_salary()
print(f"Highest paid: {highest[0]} with salary Rs. {highest[1]:,}")

print("\n2. Testing get_lowest_salary:")
lowest = database.get_lowest_salary()
print(f"Lowest paid: {lowest[0]} with salary Rs. {lowest[1]:,}")

print("\n3. Testing get_manager:")
print(f"Manager of Priya Verma: {database.get_manager('Priya Verma')}")
print(f"Manager of Rahul Sharma: {database.get_manager('Rahul Sharma')}")

print("\n4. Testing get_reports_to:")
print(f"Reporting to Rahul Sharma: {database.get_reports_to('Rahul Sharma')}")
print(f"Reporting to Priya Verma: {database.get_reports_to('Priya Verma')}")

print("\n5. Testing get_department_stats for IT:")
it_stats = database.get_department_stats("IT")
print(f"IT Stats: {it_stats}")

print("\n6. Testing get_company_stats:")
company_stats = database.get_company_stats()
print(f"Company Stats: {company_stats}")

print("\n7. Testing get_employees_by_role for Software Engineer:")
se = database.get_employees_by_role("Software Engineer")
print(f"Software Engineers: {se}")

print("\n8. Testing get_all_managers:")
managers = database.get_all_managers()
print(f"All Managers: {managers}")

print("\n9. Testing get_all_facts (sample 3):")
facts = database.get_all_facts()
print(f"Total Facts: {len(facts)}")
for f in facts[:3]:
    print(f"  - {f}")

print("\n10. Testing get_all_faq (sample 3):")
faqs = database.get_all_faq()
print(f"Total FAQs: {len(faqs)}")
for q, a in faqs[:3]:
    print(f"  - Q: {q}\n    A: {a}")

print("\n=== All Tests Finished ===")