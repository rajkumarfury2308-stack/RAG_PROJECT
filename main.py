from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

# Packages Required
# pip install langchain langchain-community langchain-text-splitters langchain-ollama langchain-chroma chromadb

# =========================================================
# LOAD DOCUMENT
# =========================================================

# loader = TextLoader(
#     "./data/Customer_Policy_Guide.txt"
# )

loader = TextLoader(
    "./Customer_Policy_Guide.txt"
)


docs = loader.load()

for doc in docs:

    doc.metadata["department"] = "customer_support"
    doc.metadata["document_type"] = "policy"
    doc.metadata["company"] = "ABC Corp"
    doc.metadata["year"] = 2026

# =========================================================
# CHUNKING
# =========================================================

# It tries to split text intelligently.
# Paragraphs
#    ↓
# Sentences
#    ↓
# Words
#    ↓
# Characters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Maximum 500 characters per chunk
    chunk_overlap=250  # Next chunk shares 250 characters 
)                     # from previous chunk

# Example : [A B C D E F G H]

# Chunk Size 4, Chunk overlap 0 -> [A B C D] [E F G H]

# Chunk Size 4, Chunk overlap 2 -> [A B C D] [C D E F] [E F G H]

splits = splitter.split_documents(docs)

# =========================================================
# EMBEDDING MODEL
# =========================================================

# converts text into mathematical vectors.
# Chunk -> Embedding Model -> vectors

# ollama pull nomic-embed-text:latest
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# =========================================================
# VECTOR STORE
# =========================================================

vector_store = Chroma(
    collection_name="customer_policy",    # equivalent to table name
    embedding_function=embeddings,        # Add Embedding Model
    persist_directory="./chroma_langchain_db",  # Stores database on disk.
)

vector_store.add_documents(splits)

# =========================================================
# RETRIEVER
# =========================================================

# Your vector DB contains:
    # many chunks
    # many embeddings

# LLM cannot read entire DB.
# Retriever finds: only the most relevant chunks

# as_retriever : Converts Chroma DB into searchable retriever object

# MMR : relevant chunks and avoid duplicate/repetitive chunks

retriever = vector_store.as_retriever(
    search_type="mmr",   # Maximal Marginal Relevance
    search_kwargs={
        "k": 2,        # Return top 2 chunks
        "fetch_k": 5   # First fetch 5 chunks
    }
)
# Flow 

# Step 1:
# Fetch top 5 similar chunks

# Step 2:
# Apply diversity filtering (MMR)

# Step 3:
# Return best 2 chunks

# Loading -> Chunking -> Embedding -> Retriever

# =========================================================
# USER QUESTION
# =========================================================

question = "Can u share the payment policy?"

# =========================================================
# RETRIEVE RELEVANT DOCS
# =========================================================

# Retriever:
    # converts question → embedding
    # searches vector DB
    # finds semantically similar chunks

retrieved_docs = retriever.invoke(question)

# =========================================================
# CREATE CONTEXT
# =========================================================

# Combines retrieved chunks into final context

docs_content = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)



# =========================================================
# LLM
# =========================================================

# brain of your RAG system

model = OllamaLLM(
    model="llama3.1",
    temperature=0.2, # randomness/creativity
    num_predict=256, # maximum output tokens
)


# Data Load -> Chunking -> Embeddings -> Retriver -> Model

# =========================================================
# PROMPT
# =========================================================

# hallucinations are controlled
# AI behavior is controlled

prompt = f"""
You are a helpful assistant for question-answering tasks.

Use ONLY the following retrieved context to answer.

If the answer is not found in the context,
say "I don't know."

Context:
{docs_content}

Question:
{question}

Answer:
"""


# =========================================================
# GENERATE RESPONSE
# =========================================================

response = model.invoke(prompt)

print("\nUser Query : ", question)
print("\nAssistant:")
print("Final Output Response: ", response)
