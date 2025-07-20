# File: rag_core.py
# This version connects to the Groq cloud service to run Llama 3 online.

import os
from groq import Groq  # Import the new library
import pandas as pd
import json

# --- Job 1: Initialize the Groq Client ---
# It's recommended to set this as an environment variable for security,
# but for this test, we'll place it directly in the code.
# ❗ Replace "YOUR_GROQ_API_KEY_HERE" with the key you got from the Groq website.
try:
    client = Groq(
        api_key="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",
    )
    print("✅ [RAG Core] Groq client initialized successfully.")
except Exception as e:
    print(f"❌ [RAG Core] Error initializing Groq client: {e}")
    client = None

# --- Job 2: Load the Knowledge Base (Unchanged) ---
try:
    df = pd.read_csv('products.csv')
    print("✅ [RAG Core] Product database loaded successfully.")
except FileNotFoundError:
    print("❌ [RAG Core] Error: 'products.csv' not found.")
    df = None

# --- Job 3: Retriever and Context Formatter (Unchanged) ---
def find_relevant_products(query, dataframe, top_k=3):
    if dataframe is None: return []
    query_words = set(query.lower().split())
    scores = []
    for index, row in dataframe.iterrows():
        product_name_words = set(row['ProductName'].lower().split())
        score = len(query_words.intersection(product_name_words))
        if score > 0:
            scores.append((score, row.to_dict()))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [product_data for score, product_data in scores[:top_k]]

def format_context_for_llm(context_data_list):
    if not context_data_list: return "هیچ محصول مرتبطی پیدا نشد."
    full_context = ""
    for i, product in enumerate(context_data_list, 1):
        full_context += f"--- محصول شماره {i} ---\n"
        try:
            specs_dict = json.loads(product.get('Specifications', '{}'))
        except (TypeError, json.JSONDecodeError):
            specs_dict = {}
        full_context += f"نام: {product.get('ProductName', 'N/A')}\n"
        full_context += f"قیمت: {int(product.get('Price', 0)):,} تومان\n"
        full_context += f"فروشگاه: {product.get('StoreName', 'N/A')}\n"
        if specs_dict:
            full_context += "مشخصات:\n"
            for key, value in specs_dict.items():
                full_context += f"  - {key}: {value}\n"
        full_context += "\n"
    return full_context.strip()

# --- Job 4: The Main RAG Logic (Updated to call Groq API) ---
def answer_with_rag(question):
    if df is None: return "Database not loaded."
    if client is None: return "Groq client not initialized. Please check your API key."

    relevant_products = find_relevant_products(question, df)
    
    if relevant_products:
        readable_context = format_context_for_llm(relevant_products)
        
        prompt = f"""
شما یک متخصص فروش بسیار باهوش و مفید در یک فروشگاه لوازم الکترونیکی هستید.
وظیفه شما این است که با استفاده از "لیست محصولات مرتبط" زیر، بهترین پاسخ یا پیشنهاد را به "سوال کاربر" ارائه دهید.
پاسخ شما باید کاملاً بر اساس اطلاعات داده شده باشد، اما آن را به صورت یک پاراگراف روان، طبیعی و محاوره‌ای به زبان فارسی بیان کنید.

[لیست محصولات مرتبط]
{readable_context}

[سوال کاربر]
{question}

[پاسخ پیشنهادی شما]
"""
        try:
            # --- THIS IS THE CRITICAL CHANGE ---
            # We now call the Groq API instead of the local Ollama.
            chat_completion = client.chat.completions.create(
                messages=[
                    # The prompt is now sent directly as the user message.
                    # The Groq API doesn't use a separate system prompt in the same way.
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192", # This is Groq's name for the Llama 3 8B model
                temperature=0.5,
                max_tokens=512,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"❌ Error calling Groq API: {e}")
            return "متاسفانه در ارتباط با سرویس آنلاین خطایی رخ داد."

    else:
        if any(word in question.lower() for word in ["سلام", "خوبی"]):
            return "سلام! من یک دستیار هوشمند محصولات هستم. چطور می‌توانم کمکتان کنم؟"
        return "متاسفانه محصولی که با درخواست شما مطابقت داشته باشد، پیدا نشد."
