# app.py - Final version matching the successful Colab RAG logic
import ollama
import pandas as pd
import json

# 1. Load the knowledge base from the local CSV file
try:
    df = pd.read_csv('products.csv')
    print("✅ Product database loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'products.csv' not found. Make sure it's in the same folder as app.py.")
    exit()

def find_relevant_product(query, dataframe):
    """
    A robust retriever that scores each product based on word matches.
    This is identical to find_relevant_product_v3 from Colab.
    """
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

def answer_with_rag(question):
    """
    The main RAG function that communicates with the local Ollama model.
    This logic now perfectly mirrors answer_with_rag_v3 from Colab.
    """
    # 1. Retrieve
    context_data = find_relevant_product(question, df)
    
    # 2. Augment & Generate
    if context_data:
        # --- CRITICAL FIX: Exactly matches Colab's JSON parsing ---
        try:
            specs_dict = json.loads(context_data['Specifications'])
            context_data['Specifications'] = specs_dict
        except (TypeError, json.JSONDecodeError):
            context_data['Specifications'] = {}
        # --- END OF FIX ---

        # Create the clean JSON string for the prompt
        context_str = json.dumps(context_data, indent=2, ensure_ascii=False)

        # Create the final user prompt with the exact same structure as Colab
        final_user_prompt = f"CONTEXT:\n```json\n{context_str}\n```\n\nQUESTION:\n{question}"

        # Use the ollama library to chat with the local model
        response = ollama.chat(
            model='phi3:mini',
            messages=[
                {'role': 'system', 'content': "You are a Q&A assistant. Answer the user's question based ONLY on the provided JSON data context. Be direct and precise."},
                {'role': 'user', 'content': final_user_prompt},
            ],
        )
        return response['message']['content']
    else:
        return "متاسفانه اطلاعاتی در مورد این محصول در دیتابیس من پیدا نشد."

# --- Main loop to run the application ---
if __name__ == '__main__':
    print("--- Local Electronics Q&A Bot (Corrected Version) ---")
    while True:
        user_question = input("\nAsk a question about a product (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        
        print("\n✅ پاسخ مدل:")
        print(answer_with_rag(user_question))