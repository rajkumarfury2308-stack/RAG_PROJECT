# RAG_PROJECT
Leave Policy Document Retrieval

# Local RAG System with LangChain, ChromaDB, and Ollama

A fully local Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, and Ollama. This project loads a company policy document, chunks it intelligently, generates vector embeddings, stores them in a local vector database, and uses a local Large Language Model (LLM) to answer user queries based **strictly** on the provided context.

## 🚀 Features
* **100% Local & Private:** No data leaves your machine. Embeddings and LLM inference run completely locally via Ollama.
* **Smart Text Chunking:** Utilizes `RecursiveCharacterTextSplitter` to maintain semantic continuity across document splits.
* **Advanced Retrieval (MMR):** Uses Maximal Marginal Relevance (MMR) to find the most relevant document chunks while eliminating redundant information.
* **Strict Context Boundary:** The LLM prompt is engineered to eliminate hallucinations by instructing the model to say "I don't know" if the answer isn't explicitly in the context.

---

## 🛠️ System Architecture

1.  **Load:** Document (`Customer_Policy_Guide.txt`) is loaded and enriched with metadata tags.
2.  **Chunk:** Text is broken down into overlapping segments (Size: 500 chars, Overlap: 250 chars).
3.  **Embed:** Chunks are converted into mathematical vectors using the `nomic-embed-text` model.
4.  **Store:** Vectors are saved locally onto the disk inside a `Chroma` vector database.
5.  **Retrieve:** A multi-step MMR search fetches the top 5 relevant chunks and filters down to the best 2 diverse chunks.
6.  **Generate:** The context and user query are fed into `llama3.1` to generate a precise, grounded answer.

---

## 📦 Prerequisites & Installation

1. Install Ollama and Models
Ensure you have [Ollama](https://ollama.com/) installed on your machine. Once installed, pull the required embedding and LLM models via your terminal:
# Pull the embedding model
ollama pull nomic-embed-text:latest


2. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME


3. Install Dependencies
Install the required Python packages using pip:
pip install langchain langchain-community langchain-text-splitters langchain-ollama langchain-chroma chromadb

Project Structure:
├── data/
│   └── Customer_Policy_Guide.txt   # Your source policy document
├── chroma_langchain_db/            # Local vector store directory (auto-generated)
├── main.py                         # Core application script
└── README.md                       # Project documentation

💻 Usage
Place your target text document inside the data/ folder and name it Customer_Policy_Guide.txt.
Run the main execution script:
python main.py

Example Output:
User Query :  Can u share the payment policy?
Assistant:
Final Output Response:  According to the provided customer policy guide, all payments must be processed by the 5th of every month. Late payments will incur a 5% penalty fee...

⚙️ Configuration Details
---

## ⚙️ Configuration Details

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Chunk Size** | `500` | Maximum character limit per text split. |
| **Chunk Overlap** | `250` | Number of shared characters between adjacent chunks to maintain context. |
| **Search Type** | `mmr` | Maximal Marginal Relevance for maximizing query relevance and chunk diversity. |
| **k / fetch_k** | `2 / 5` | Fetches 5 candidate chunks, outputs the top 2 optimal results. |
| **Temperature** | `0.2` | Keeps the model focused, deterministic, and factual. |



# Pull the LLM model
ollama pull llama3.1
