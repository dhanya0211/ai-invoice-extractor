# AI Invoice Analyzer & RAG Chatbot (Local CPU Edition)

An entirely local, CPU-friendly AI application that extracts data from invoices (PDFs/Images) using OCR, parses the data into structured JSON using a local HuggingFace LLM (Qwen), and allows users to "chat" with their past invoices using a custom SQLite Vector Search (RAG).

##  Features
* **Completely Local & Private:** No data is sent to OpenAI or cloud APIs. Everything runs on your own machine.
* **Hybrid OCR:** Uses `PyMuPDF` for digital PDFs and `EasyOCR` as a fallback for scanned images.
* **Smart Data Parsing:** Uses `Qwen2.5-0.5B-Instruct` to reliably convert raw invoice text into clean JSON.
* **Custom Vector Database:** Uses standard `SQLite3` combined with `NumPy` Cosine Similarity to search for related invoices mathematically.
* **Interactive Chatbot:** Ask questions like *"How much did I spend at Walmart?"* and the AI will answer based on your database history.

##  Project Structure
* `main.py` - The main FastAPI application (Upload & Chat endpoints).
* `database_connection.py` - Handles SQLite table creation, saving BLOB embeddings, and Vector Math (Cosine Similarity).
* `embedding.py` - Generates mathematical text embeddings for the AI.
* `invoices.db` - The local SQLite database automatically created on the first run.

##  Installation & Setup

1. **Install Python dependencies:**
   Make sure you have Python 3.9+ installed, then run:
   ```bash
   git clone https://github.com/dhanya0211/ai-invoice-analyzer.git
   cd ai-invoice-analyzer
   pip install -r requirements.txt
   
   # Or install packages manually if you prefer:
   pip install fastapi uvicorn python-multipart pydantic PyMuPDF Pillow numpy easyocr transformers torch torchvision
Run the Application:
Start the FastAPI server using Uvicorn:
code
Bash
uvicorn main:app --reload
First-Run Model Downloads:
On the very first execution, the application will automatically download the required AI models to your local machine:
Qwen2.5-0.5B-Instruct (for JSON parsing and Chat generation - approx. 1-2GB)
EasyOCR Weights (for image text extraction)
Embedding Model (for vectorizing text, e.g., standard HuggingFace sentence-transformers)
(Note: Please be patient during the initial startup as these downloads may take a few minutes depending on your internet connection).

Usage

Once the server is running, you can access the automatically generated Swagger UI to test and interact with the API directly from your browser:
http://127.0.0.1:8000/docs
1. Uploading an Invoice (POST /upload)
Use the /upload endpoint to submit an invoice file (.pdf, .png, .jpg).
What happens behind the scenes:
PyMuPDF attempts to extract text. If the file is a scanned image, EasyOCR automatically takes over as a fallback.
The raw text is sent to the local Qwen LLM, which smartly extracts key data (Vendor, Date, Total Amount, Items) into a structured JSON format.
The JSON data is converted into a vector embedding and stored as a binary BLOB in invoices.db alongside the parsed text.
2. Chatting with Your Data (POST /chat)
Use the /chat endpoint to ask natural language questions about your uploaded invoices. Example payload:
code
JSON
{
  "query": "How much did I spend at Walmart last month?"
}
What happens behind the scenes:
Your question is converted into a mathematical vector embedding.
database_connection.py pulls the saved vectors from SQLite and calculates the Cosine Similarity using NumPy.
The most relevant invoices are retrieved (RAG - Retrieval-Augmented Generation) and passed to the Qwen LLM as context.
The LLM generates a conversational answer based only on your actual invoice history.

How the Custom Vector Database Works

Instead of relying on heavy, memory-hungry vector databases like ChromaDB, Milvus, or FAISS, this project implements a lightweight alternative using standard SQLite3:
Text embeddings (arrays of floats) are packed into raw bytes and saved in a standard SQLite column.
During a search, the arrays are unpacked from the database into NumPy arrays, and mathematically compared against the search query using the Cosine Similarity formula:
Similarity = (A · B) / (||A|| * ||B||)
This allows for powerful semantic search with zero extra background services or Docker containers!

 Hardware Requirements

Because this is the Local CPU Edition, it is heavily optimized for standard computers without dedicated GPUs:
RAM: 8GB minimum (16GB recommended for faster LLM inference and smooth OCR).
Storage: ~3GB to 4GB of free space for the model weights and database.
CPU: Any modern multi-core processor (Intel Core i5/i7, AMD Ryzen, or Apple Silicon).
🛠️ Troubleshooting
Slow processing times? EasyOCR can be resource-intensive on a CPU. Try uploading native digital PDFs instead of scanned images whenever possible.
LLM Responses are cut off? You may need to adjust the max_new_tokens parameter in the main.py transformer pipeline configuration.
Out of Memory Error? Ensure you don't have other heavy applications running. While the Qwen 0.5B model is tiny by LLM standards, holding OCR models, embedding models, and LLMs in RAM simultaneously requires some breathing room.

License

MIT License - Free to use, modify, and distribute for personal or commercial projects.