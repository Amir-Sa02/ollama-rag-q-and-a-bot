# Local RAG Q&A Bot for Electronics Catalog

This project is a fully local and offline-capable Question & Answer chatbot designed to act as an intelligent sales assistant for an electronics product catalog. It leverages a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based on a local `products.csv` knowledge base.

The application features a web-based chat interface built with Flask and is powered by a local `Llama 3.1 8B` model running on Ollama.

---

## Core Features

* **Fully Local & Offline:** The entire system, including the Large Language Model, runs on a local machine with no internet connection required.
* **RAG Architecture:** Ensures answers are factually grounded in the provided product data, preventing model hallucination.
* **Advanced Conversational AI:**
    * **Multi-Product Retrieval:** Capable of retrieving and reasoning about multiple relevant products to answer complex comparison and recommendation questions.
    * **Chat History:** Maintains conversation context for natural, follow-up interactions.
    * **Behavioral Guardrails:** Utilizes an advanced system prompt to define the bot's persona ("Tek-yar"), enforce strict rules of engagement, and ensure responses remain professional and on-topic.
* **Web Interface:** A clean and user-friendly chat interface built with Flask, HTML, and JavaScript.

---

## Technical Architecture

The system is composed of three primary components that work together locally:

1.  **Knowledge Base:**
    * A single `products.csv` file acts as the "source of truth". It contains structured data about various electronic products, including names, prices, stores, and detailed specifications in a nested JSON format.

2.  **Retriever:**
    * Implemented in the `find_relevant_products` function within `rag_core.py`.
    * This is a keyword-based retriever that scores every product in the CSV against the user's query based on the number of overlapping words.
    * It returns the **top-k** (default is 3) most relevant products, enabling the system to handle complex queries.

3.  **Generator:**
    * **LLM:** The system uses the `llama3.1:8b` model, served locally via the **Ollama** framework.
    * **Prompt Engineering:** A sophisticated system prompt defines the bot's persona and establishes strict behavioral guardrails. It instructs the model to adhere strictly to the provided context, maintain its area of expertise, and avoid inventing information.
    * **Context Formatting:** The `format_context_for_llm` function translates the raw data retrieved from the CSV into a human-readable, Persian paragraph. This pre-processing step makes the context significantly easier for the LLM to understand and reason about, leading to higher-quality responses.

---

## How to Run Locally

1.  **Prerequisites:**
    * Python 3.x
    * Ollama installed and running.
    * Git

2.  **Setup:**
    * Clone the repository:
        ```bash
        git clone [https://github.com/Amir-Sa02/ollama-rag-q-and-a-bot.git](https://github.com/Amir-Sa02/ollama-rag-q-and-a-bot.git)
        cd ollama-rag-q-and-a-bot
        ```
    * Install the required Python libraries:
        ```bash
        pip install flask pandas ollama
        ```
    * Download the required LLM via Ollama:
        ```bash
        ollama pull llama3.1:8b
        ```
    * Ensure your `products.csv` file is present in the root directory.

3.  **Execution:**
    * Run the Flask web server from your terminal:
        ```bash
        python app.py
        ```
    * Open your web browser and navigate to **`http://127.0.0.1:5000/`** to start chatting.

---

## Identified Limitations & Future Work

This version of the project, while powerful, has clear areas for future improvement:

* **Keyword-Based Retrieval:** The current retriever is based on simple keyword matching. It cannot understand the semantic meaning or intent behind a query (e.g., "a good laptop for programming"). The next major step is to upgrade this component to a **semantic search** system using a dedicated embedding model and a vector database.
* **Limited Data Formats:** The system is hard-coded to read a single, structured `products.csv` file. It cannot ingest unstructured data like PDFs or text from web pages.
* **Manual Memory Management:** The chat history is passed manually between the web server and the core logic.

The logical next step in this project's evolution is to integrate a dedicated RAG framework like **LlamaIndex**, which would solve all of the above limitations by providing built-in support for semantic search, multi-format data ingestion, and optimized memory management.