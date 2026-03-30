import sqlite3
import numpy as np

# 1. INITIALIZE DATABASE
def init_db():
    conn = sqlite3.connect('invoices.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor TEXT,
                    invoice_number TEXT,
                    invoice_date TEXT,
                    due_date TEXT,
                    total_amount REAL,
                    currency TEXT,
                    embedding BLOB,
                    raw_json TEXT
    )''')
    conn.commit()
    conn.close()

# Run initialization when this file is imported
init_db()

# 2. SAVE INVOICE FUNCTION
def save_invoice_to_db(vendor, invoice_number, invoice_date, due_date, total_amount, currency, embedding, raw_json):
    """Saves the extracted invoice data and its AI embedding to SQLite."""
    conn = sqlite3.connect('invoices.db')
    cursor = conn.cursor()
    
    # Convert the embedding list/array into a raw binary format (BLOB) for SQLite
    embedding = np.array(embedding, dtype=np.float32).tobytes()
    
    cursor.execute('''
        INSERT INTO invoices 
        (vendor, invoice_number, invoice_date, due_date, total_amount, currency, embedding, raw_json) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (vendor, invoice_number, invoice_date, due_date, total_amount, currency, embedding, raw_json))
    
    conn.commit()
    conn.close()
    print(f"Saved invoice from {vendor} to DB successfully.")

# 3. SEARCH FUNCTION (FOR THE CHATBOT)
def search_similar_invoices(query_embedding, top_k=3):
    """Finds the most relevant invoices by comparing embeddings mathematically."""
    conn = sqlite3.connect('invoices.db')
    cursor = conn.cursor()
    
    # Fetch all records that have an embedding
    cursor.execute('SELECT raw_json, embedding FROM invoices WHERE embedding IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return[]

    # Ensure query is a numpy array
    query_vec = np.array(query_embedding, dtype=np.float32)
    
    results =[]
    
    for row in rows:
        raw_json = row[0]
        embedding_blob = row[1]
        
        # Convert the BLOB back into a usable Python math array
        db_vec = np.frombuffer(embedding_blob, dtype=np.float32)
        
        # --- COSINE SIMILARITY MATH ---
        # This checks how similar the user's question is to this specific invoice
        norm_query = np.linalg.norm(query_vec)
        norm_db = np.linalg.norm(db_vec)
        
        if norm_query == 0 or norm_db == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_vec, db_vec) / (norm_query * norm_db)
            
        results.append((similarity, raw_json))
        
    # Sort the results so the highest similarity score is at the top
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Extract just the raw_json strings for the top 'k' matches
    top_matches = [match[1] for match in results[:top_k]]
    
    return top_matches