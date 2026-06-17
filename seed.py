from database import init_db, add_employee

init_db()

add_employee("Rahul Sharma", "CEO", "Management", 200000, None)
add_employee("Priya Verma", "CTO", "IT", 180000, "Rahul Sharma")
add_employee("Neha Gupta", "HR Manager", "HR", 120000, "Rahul Sharma")
add_employee("Amit Kumar", "Software Engineer", "IT", 70000, "Priya Verma")
add_employee("Rohit Singh", "Sales Executive", "Sales", 60000, "Rahul Sharma")
add_employee("Ananya Jain", "Marketing Manager", "Marketing", 75000, "Rahul Sharma")
add_employee("Karan Mehta", "Financial Analyst", "Finance", 80000, "Rahul Sharma")
add_employee("Sneha Patel", "Operations Manager", "Operations", 85000, "Rahul Sharma")

print("Database seeded!")