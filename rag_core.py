# File: rag_core.py
# Final version with the most direct prompt possible for maximum consistency.

import ollama
import pandas as pd
import json

# --- Job 1: Load the Knowledge Base (Unchanged) ---
try:
    df = pd.read_csv('products.csv')
    print("✅ [RAG Core] Product database loaded successfully.")
except FileNotFoundError:
    print("❌ [RAG Core] Error: 'products.csv' not found.")
    df = None

# --- Job 2: The Retriever Function (Unchanged) ---
def find_relevant_product(query, dataframe):
    query_words = set(query.lower().split())
    best_match = None
    highest_score = 0
    for index, row in dataframe.iterrows():
        product_name_words = set(row['ProductName'].lower().split())
        score = len(query_words.intersection(product_name_words))
        if score > highest_score:
            highest_score = score
            best_match = row
    if highest_score > 1:
        return best_match.to_dict()
    return None

# --- Helper function to format context (Unchanged) ---
def format_context_for_llm(context_data):
    try:
        specs_dict = json.loads(context_data.get('Specifications', '{}'))
    except (TypeError, json.JSONDecodeError):
        specs_dict = {}
    info = f"اطلاعات محصول: \n"
    info += f"- نام: {context_data.get('ProductName', 'N/A')}\n"
    info += f"- قیمت: {int(context_data.get('Price', 0)):,} تومان\n"
    info += f"- فروشگاه: {context_data.get('StoreName', 'N/A')}\n"
    if specs_dict:
        info += "- مشخصات:\n"
        for key, value in specs_dict.items():
            info += f"  - {key}: {value}\n"
    return info.strip()

# --- Job 3: The Main RAG Logic (Final Prompt Engineering) ---
def answer_with_rag(question):
    if df is None:
        return "Database not loaded."

    context_data = find_relevant_product(question, df)
    
    if context_data:
        readable_context = format_context_for_llm(context_data)
        
        # --- THE FINAL, MOST DIRECT PROMPT ---
        prompt = f"""[اطلاعات]
{readable_context}

[سوال]
{question}

با توجه به اطلاعات بالا، به سوال به صورت کوتاه و مستقیم پاسخ بده.
[پاسخ]
"""
        
        response = ollama.generate(
            model='phi3:mini',
            prompt=prompt,
            options={
                'temperature': 0.1,
                'stop': ['\n'] # Stop generating after the first line break.
            }
        )
        return response['response'].strip()
    
    else:
        if any(word in question.lower() for word in ["سلام", "خوبی"]):
            return "سلام! من یک دستیار هوشمند محصولات هستم. چطور می‌توانم کمکتان کنم؟"
        return "متاسفانه اطلاعاتی در مورد این محصول در دیتابیس من پیدا نشد."