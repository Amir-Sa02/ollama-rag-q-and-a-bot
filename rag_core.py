import ollama  # The official library to communicate with the local Ollama service.
import pandas as pd  # The standard library for data manipulation, used here to read and search the CSV.
import json  # Used for parsing the nested JSON string in the 'Specifications' column.

# --- Job 1: Load the Knowledge Base ---
# This is the "Retrieval" part of RAG. We load the entire knowledge base into memory.
try:
    df = pd.read_csv('products.csv')
    print("Product database loaded successfully.")
except FileNotFoundError:
    print("Error: 'products.csv' not found. Make sure it's in the same folder as app.py.")
    exit()

# --- Job 2: The Retriever Function ---
# This function acts as our simple, keyword-based search engine.
def find_relevant_product(query, dataframe):
    """
    A robust retriever that scores each product based on word matches.
    Its goal is to find the single most relevant document (product row) from our knowledge base.
    """
    query_words = set(query.lower().split())
    best_match = None
    highest_score = 0

    # This loop iterates through every row in our database (the DataFrame).
    for index, row in dataframe.iterrows():
        product_name_words = set(row['ProductName'].lower().split())
        # It calculates a simple relevance 'score' by counting shared words.
        score = len(query_words.intersection(product_name_words))
        
        # If the current product is a better match than the previous best, we update it.
        if score > highest_score:
            highest_score = score
            best_match = row

    # We only return a result if the match is reasonably confident (score > 1).
    # This prevents returning irrelevant products for vague questions.
    if highest_score > 1:
        return best_match.to_dict()
    return None

# --- Job 3: The Main RAG Logic ---
# This function orchestrates the entire RAG process.
def answer_with_rag(question):
    """
    The main RAG function that communicates with the local Ollama model.
    It combines the Retrieval and Generation steps.
    """
    # --- Step 1: RETRIEVAL ---
    # Call our search engine to find the relevant context for the user's question.
    context_data = find_relevant_product(question, df)
    
    # --- Step 2: AUGMENTATION & GENERATION ---
    # This block only runs if our retriever successfully found a relevant product.
    if context_data:
        # --- CRITICAL DATA CLEANING ---
        # This is the fix that made the model work correctly. The 'Specifications'
        # column is a string that looks like a JSON object. We must parse it into
        # a real dictionary so it can be cleanly formatted in the prompt.
        try:
            specs_dict = json.loads(context_data['Specifications'])
            context_data['Specifications'] = specs_dict
        except (TypeError, json.JSONDecodeError):
            # If parsing fails, just leave it as is to prevent a crash.
            context_data['Specifications'] = {}
        # --- END OF FIX ---

        # Create a clean, multi-line JSON string. This structured format is much
        # easier for the LLM to read and understand than a messy, single-line string.
        # json.dumps() is a function in Python’s json module that converts a Python object into a JSON formatted string.
        # It allows you to serialize Python objects such as dictionaries, lists, and more into JSON format.
        context_str = json.dumps(context_data, indent=2, ensure_ascii=False)

        # --- PROMPT ENGINEERING ---
        # This is where we "augment" the user's query. 
        # We construct a detailed prompt to guide the LLM's behavior, telling it exactly what to do.
        final_user_prompt = f"CONTEXT:\n```json\n{context_str}\n```\n\nQUESTION:\n{question}"

        # --- GENERATION ---
        # We send the request to the local Ollama model.
        response = ollama.chat(
            model='phi3:mini',  # Specifies which model to use.
            messages=[
                # The System Prompt sets the rules and persona for the AI.
                # This is a key part of RAG, ensuring the model doesn't use its general knowledge.
                {'role': 'system', 'content': "You are a Q&A assistant. Answer the user's question based ONLY on the provided JSON data context in Persian. Be direct and precise."},
                # The User Prompt contains the context we found and the original question.
                {'role': 'user', 'content': final_user_prompt},
            ],
        )
        # The ollama.chat() function returns a dictionary object with a standard structure.
        # The 'response' object looks something like this:
        # {
        #   'model': 'phi3:mini',
        #   'message': {
        #     'role': 'assistant',
        #     'content': 'This is the actual text answer from the model.'
        #   },
        #   ... and other metadata ...
        # }
        # Therefore, response['message']['content'] is the standard way to access
        # the final text string generated by the assistant.
        return response['message']['content']
    else:
        # If the retriever found no relevant product, return a standard "not found" message.
        return "متاسفانه اطلاعاتی در مورد این محصول در دیتابیس من پیدا نشد."


