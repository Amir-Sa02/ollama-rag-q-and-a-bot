import ollama
import pandas as pd
import json

# Model Configuration
# We now point to the local model downloaded with Ollama.
MODEL_TO_USE = "llama3.1:8b"

# Load Knowledge Base
try:
    df = pd.read_csv('products.csv')
    print("✅ [RAG Core] Product database loaded.")
except FileNotFoundError:
    print("❌ [RAG Core] Error: 'products.csv' not found.")
    df = None

# Retriever and Formatter
def find_relevant_products(query, dataframe, top_k=3):
    """
    Finds the top K most relevant products from the dataframe based on a user query.
    This acts as the "Retriever" in our RAG system.

    Args:
        query (str): The user's question.
        dataframe (pd.DataFrame): The DataFrame containing all product data.
        top_k (int): The maximum number of relevant products to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a relevant product.
    """
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
    """
    Converts a list of product dictionaries into a single, human-readable Persian string.
    This function acts as a "translator" for the LLM, making the context much easier to understand
    than raw JSON, which prevents confusion and improves response quality.

    Args:
        context_data_list (list): A list of product dictionaries retrieved from the database.

    Returns:
        str: A formatted, multi-line string containing the product information in Persian.
    """   
    if not context_data_list: return ""
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

# Main RAG Logic (with Guardrails)
def answer_with_rag(question, history):
    if df is None: return "Database not loaded."

    # --- Step 1: Retrieval ---
    relevant_products = find_relevant_products(question, df)
    readable_context = format_context_for_llm(relevant_products)

    # --- Step 2: Advanced Prompt Engineering with Guardrails ---
    system_prompt = """
شما یک دستیار هوش مصنوعی هستید که متخصص فروش لوازم الکترونیکی است و دارای اطلاعات مربوط به لوازم الکترونیکی در اکثر فروشگاه‌های آنلاین ایرانی می‌باشد. شخصیت شما خوش‌برخورد، مفید و کاملاً حرفه‌ای است.

**قوانین رفتاری شما (بسیار مهم):**
1.  **حفظ حوزه تخصصی:** شما فقط و فقط در مورد محصولات الکترونیکی داخل دیتاست صحبت می‌کنید. اگر سوالی خارج از این حوزه پرسیده شد (مانند ادبیات، تاریخ، حیوانات و...)، با احترام پاسخ دهید که تخصص شما در این زمینه نیست و گفتگو را به سمت محصولات الکترونیکی هدایت کنید.
2.  **پایبندی کامل به منبع:** پاسخ‌های شما باید **همیشه و فقط** بر اساس "اطلاعات محصول مرتبط" که در ادامه ارائه می‌شود، باشد. **هرگز، تحت هیچ شرایطی، محصولی را از خود ابداع نکنید.** اگر اطلاعاتی در منبع وجود نداشت، به صراحت بگویید که آن اطلاعات را در اختیار ندارید.
3.  **رفتار محاوره‌ای:** با کاربران به صورت دوستانه صحبت کنید. اگر کاربر جوک گفت یا تشکر کرد، پاسخ مناسبی بدهید، اما همیشه در نهایت به وظیفه اصلی خود به عنوان دستیار خرید برگردید.
4.  **استفاده از تاریخچه:** از تاریخچه مکالمه برای درک بهتر سوالات بعدی کاربر استفاده کنید تا مکالمه‌ای طبیعی و منسجم داشته باشید.
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    
    user_content = f"سوال کاربر: {question}"
    if readable_context:
        user_content += f"\n\n[اطلاعات محصول مرتبط]\n{readable_context}"
    
    messages.append({"role": "user", "content": user_content})

    try:
        # We now call the local Ollama service instead of the Groq API.
        response = ollama.chat(
            messages=messages,
            model=MODEL_TO_USE,
            options={
                'temperature': 0.2
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Error calling local Ollama model: {e}")
        return "در ارتباط با سرویس محلی خطایی رخ داد"
