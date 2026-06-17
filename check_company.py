import sqlite3

conn = sqlite3.connect("company.db")
cur = conn.cursor()

cur.execute("SELECT * FROM company_info")

rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()