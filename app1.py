from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
import fitz  # PyMuPDF
from typing import Optional, Union, List
import io
import numpy as np
from PIL import Image
import json
from embeddings import get_embeddings  
from database_connection import save_invoice_to_db, search_similar_invoices 

# ------------------------- 
# Local AI / Transformer Imports
# -------------------------
import easyocr
from transformers import pipeline

print("Loading OCR Model...")
ocr_reader = easyocr.Reader(['en'], gpu=False)

print("Loading LLM for parsing & chatting...")
llm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    device="cpu" 
)
print("Models loaded successfully!")

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="AI Invoice RAG Chatbot")
app.mount("/static", StaticFiles(directory="."), name="static")

# -------------------------
# Pydantic Schemas
# -------------------------
class InvoiceData(BaseModel):
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: Union[str, float, None] = None
    currency: Optional[str] = "USD"
    valid: bool = True

# New Schema for the Chatbot
class ChatRequest(BaseModel):
    user_message: str
    # Optional: You can add history here later if you want a multi-turn conversation
    # history: List[dict] =[] 

class ChatResponse(BaseModel):
    reply: str
    sources_used: List[str]

# -------------------------
# Helpers
# -------------------------
def extract_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    results = ocr_reader.readtext(img_np, detail=0)
    return "\n".join(results)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc: text += page.get_text()
    text = text.strip()
    if not text:
        pix = doc[0].get_pixmap()
        text = extract_text_from_image(pix.tobytes("png"))
    return text

def extract_json_from_text(raw_text):
    if not raw_text: return "{}"
    start_idx = raw_text.find('{')
    end_idx = raw_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        return raw_text[start_idx:end_idx+1]
    return raw_text.strip()

# -------------------------
# 1. INGESTION ENDPOINT
# -------------------------
@app.post("/analyze-invoice", response_model=InvoiceData)
def analyze_invoice(file: UploadFile = File(...)):
    """Extracts data from file, gets real embedding, saves to DB"""
    file_bytes = file.file.read()

    try:
        # Extract Text
        if file.content_type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        else:
            extracted_text = extract_text_from_image(file_bytes)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="No readable text.")

        # Parse with LLM
        system_prompt = """Extract details into this exact JSON schema:
        {"vendor": "name", "invoice_number": "num", "invoice_date": "YYYY-MM-DD", "due_date": "YYYY-MM-DD", "total_amount": "num", "currency": "USD", "valid": true}
        Return ONLY valid JSON."""

        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"INVOICE TEXT:\n{extracted_text}"}
        ]

        output = llm_pipeline(messages, max_new_tokens=200, temperature=0.0, do_sample=False)
        raw_response = output[0]["generated_text"][-1]["content"]
        
        # Validate JSON
        clean_json_string = extract_json_from_text(raw_response)
        try:
            invoice = InvoiceData.model_validate_json(clean_json_string)
        except ValidationError:
            raise HTTPException(status_code=422, detail="AI failed to generate valid JSON.")

        # ---------------------------------------------------------
        # INTEGRATE YOUR FILES HERE
        # ---------------------------------------------------------
        invoice_dict = invoice.model_dump()
        
        # 1. Create a descriptive string to embed (combining the JSON + raw text context)
        text_to_embed = f"Vendor: {invoice.vendor}, Amount: {invoice.total_amount} {invoice.currency}, Date: {invoice.invoice_date}. Raw Text: {extracted_text[:300]}"
        
        # 2. Get real embedding from your file
        embedding_vector = get_embeddings(text_to_embed) 
        
        # 3. Save using your database connection file
        # Inside def analyze_invoice(...):

        save_invoice_to_db(
            vendor=invoice.vendor,
            invoice_number=invoice.invoice_number,
            invoice_date=invoice.invoice_date,
            due_date=invoice.due_date,
            total_amount=invoice.total_amount,
            currency=invoice.currency,
            embedding=embedding_vector, # This comes from your embedding.py
            raw_json=json.dumps(invoice.model_dump())
        )
        # ---------------------------------------------------------
        
        return invoice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# -------------------------
# 2. CHATBOT ENDPOINT (NEW)
# -------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_with_invoices(chat_request: ChatRequest):
    """Answers user questions based on database history."""
    user_query = chat_request.user_message

    try:
        # 1. Embed the user's question
        query_embedding = get_embeddings(user_query)

        # 2. Search your database for the Top 3 most relevant invoices
        # (Your DB file should return a list of JSON strings or dictionaries)
        top_matches = search_similar_invoices(query_embedding, top_k=3)

        # 3. Format the retrieved data into a string "Context"
        context_text = "\n\n".join([f"Invoice {i+1}: {match}" for i, match in enumerate(top_matches)])
        
        # 4. Formulate the Chatbot Prompt
        system_prompt = f"""You are a helpful Financial Assistant Chatbot. 
        Answer the user's question using ONLY the provided database context below. 
        If the answer is not in the context, say "I don't have records of that in the database."
        
        DATABASE CONTEXT:
        {context_text}
        """

        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # 5. Generate Answer using Qwen
        output = llm_pipeline(
            messages, 
            max_new_tokens=250, 
            temperature=0.3, # Slight temperature makes it sound more conversational
            do_sample=True
        )
        bot_reply = output[0]["generated_text"][-1]["content"]

        # 6. Return response
        return ChatResponse(
            reply=bot_reply,
            sources_used=[f"Retrieved {len(top_matches)} related invoices from DB."]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")