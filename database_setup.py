import sqlite3
conn = sqlite3.connect('invoices.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS invoices (
               id INTEGER PRIMARY KEY AUTOINCREMENT ,
                vendor TEXT,
                invoice_number TEXT,
                invoice_date TEXT,
                due_date TEXT,
                total_amount REAL,
                currency TEXT,
               raw_json TEXT
)''')
conn.commit()
