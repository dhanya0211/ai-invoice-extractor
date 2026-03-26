import os
import json                  # 1. Imports moved to the very top
import PIL.Image             # Required to open the invoice image
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
# Load the .env file and set up the API key
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 2. Define the model
model = genai.GenerativeModel('gemini-2.5-flash')
def save_database(data_json):
    try:
        conn = sqlite3.connect('invoices.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO invoices (vendor, invoice_number, invoice_date, due_date, total_amount, currency, raw_json) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                        (data_json['vendor'], data_json['invoice_number'], data_json['invoice_date'], 
                        data_json['due_date'], data_json['total_amount'], data_json['currency'], json.dumps(data_json)))
        conn.commit()
        conn.close()
        print("Data saved to database successfully!")
    except Exception as e:
        print(f"Error saving to database: {e}")

def clean_ai_json(raw_text):
    """Safely removes markdown and returns clean JSON string."""
    if not raw_text:
        return ""
    
    # 1. Start with the raw text
    cleaned = raw_text.strip()
    
    # 2. If it contains backticks, surgically extract the middle
    if "```" in cleaned:
        # Split by ``` and take the second part (the content)
        # Then split that by ``` again and take the first part
        parts = cleaned.split("```")
        for part in parts:
            # Look for the part that looks like JSON (starts with {)
            if "{" in part:
                cleaned = part.replace("json", "").strip()
                break
                
    return cleaned
    

try:
    # 3. Define the image and prompt BEFORE calling the model
    # (Make sure "invoice.jpg" is actually in your folder!)
    image = PIL.Image.open(r"C:\Users\dhanya.s\Desktop\github_practice\data\invoice.webp") 
    prompt = "Extract the vendor, invoice number, invoice date, due date, total amount, and currency from this invoice. Return the data in JSON format with the following keys: vendor, invoice_number, invoice_date, due_date, total_amount, currency.Provide the exact result as JSON."
    
    # Process the invoice
    print("Sending to AI...")
    response = model.generate_content([prompt, image])
    print("\n--- AI Result ---")
    raw_text = response.text.strip()
    result = clean_ai_json(raw_text)
    print(f"Raw AI Response: {result}")
    if not result.strip():
        raise ValueError("AI response is empty. No data extracted.")
    else:
        json_data = json.loads(result)

        save_database(json_data)

        # 4. Save the ACTUAL response from the AI into the JSON file
        with open("extracted_data.json", "w") as f:
            f.write(response.text)
            
        print("\nSuccess! Data saved to extracted_data.json")
    # Error Handling
except FileNotFoundError:
    print("Error: Could not find the invoice image file. Is it named correctly?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")