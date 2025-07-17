# Local Q&A Bot for Electronics using RAG and Ollama

This project is a local, offline-capable Question and Answer bot that uses a Retrieval-Augmented Generation (RAG) architecture to answer questions about a catalog of electronic products. The system uses a local `products.csv` file as its knowledge base and leverages the `phi3:mini` model running on Ollama to generate answers.

## Project Journey & Key Decisions

This project did not start with RAG. The final architecture was chosen after initial attempts to use traditional fine-tuning proved to be unstable and unreliable for this specific use case.

### Initial Goal: Fine-Tuning

The original plan was to fine-tune an open-source model (`Phi-3-mini`) on a Persian dataset of product Q&A pairs. The goal was to "teach" the model the facts from our product catalog.

### The Challenge: Fine-Tuning Failures

Multiple attempts at fine-tuning led to significant issues:
1.  **Factual Inaccuracy:** Initial training runs (e.g., 1 epoch) resulted in a model that understood the *format* of an answer but produced factually incorrect information (e.g., wrong prices or specs).
2.  **Model Collapse / Overfitting:** Attempts to improve accuracy by training for more epochs or using different learning rates resulted in a worse outcome. The model became over-specialized on the small training set and lost its fundamental ability to generate coherent language, producing nonsensical answers.

This led to the conclusion that **fine-tuning is not the right tool for teaching a model a set of specific, changing facts**. It is better suited for teaching a model a new *behavior* or *style*.

### The Pivot: Adopting the RAG Architecture

Faced with the limitations of fine-tuning, the project's strategy was fundamentally changed to a **Retrieval-Augmented Generation (RAG)** approach.

**Why RAG was the solution:**
* **Accuracy:** Answers are generated based on context retrieved directly from the `products.csv` file, eliminating hallucinations and ensuring factual correctness.
* **Maintainability:** Updating the product catalog is as simple as editing the CSV file. No complex model retraining is required.
* **Stability:** We use the powerful, pre-trained base model as-is, avoiding the entire unstable and resource-intensive fine-tuning process.

## Architecture of the Final Solution

The final system consists of three main components running locally:
1.  **Knowledge Base:** A simple `products.csv` file containing all product information.
2.  **Retriever:** A Python function within `app.py` that searches the CSV to find the product most relevant to the user's query.
3.  **Generator:** The base `phi3:mini` model running on **Ollama**, which receives the retrieved information and the user's question, and generates a natural language answer.

## How to Run Locally

1.  **Prerequisites:**
    * Python 3.x installed.
    * Ollama installed and running.

2.  **Setup:**
    * Clone this repository.
    * Install the required Python libraries:
        ```bash
        pip install pandas ollama
        ```
    * Pull the `phi3:mini` model:
        ```bash
        ollama pull phi3:mini
        ```
    * Ensure the `products.csv` file is in the same directory as `app.py`.

3.  **Execution:**
    * Run the application from your terminal:
        ```bash
        python app.py
        ```
    * Start asking questions! Type `exit` to close the program.

## Future Work

To handle a much larger dataset (e.g., >100,000 products), the simple CSV search function could be upgraded to a more efficient **Vector Database** (e.g., using FAISS or ChromaDB) for near-instantaneous semantic retrieval.
