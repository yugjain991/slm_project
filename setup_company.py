import sqlite3

conn = sqlite3.connect("company.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS company_info(
    key TEXT PRIMARY KEY,
    value TEXT
)
""")

data = [
    ("CEO", "Rahul Sharma"),
    ("Company", "BrightTech Solutions"),
    ("Location", "Noida"),
    ("Founded", "2020")
]

for row in data:
    cur.execute(
        "INSERT OR REPLACE INTO company_info VALUES (?, ?)",
        row
    )

conn.commit()
conn.close()

print("Company information added successfully!")