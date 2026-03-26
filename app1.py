import os
import json                  # 1. Imports moved to the very top
import PIL.Image             # Required to open the invoice image
import google.generativeai as genai
from dotenv import load_dotenv

# Load the .env file and set up the API key
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 2. Define the model
model = genai.GenerativeModel('gemini-2.5-flash')

try:
    # 3. Define the image and prompt BEFORE calling the model
    # (Make sure "invoice.jpg" is actually in your folder!)
    image = PIL.Image.open(r"C:\Users\dhanya.s\Desktop\github_practice\data\invoice.webp") 
    prompt = "Extract the invoice number, date, and total amount. Provide the exact result as JSON."
    
    # Process the invoice
    print("Sending to AI...")
    response = model.generate_content([prompt, image])
    print("\n--- AI Result ---")
    print(response.text)

    # 4. Save the ACTUAL response from the AI into the JSON file
    with open("extracted_data.json", "w") as f:
        f.write(response.text)
        
    print("\nSuccess! Data saved to extracted_data.json")

# Error Handling
except FileNotFoundError:
    print("Error: Could not find the invoice image file. Is it named correctly?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")