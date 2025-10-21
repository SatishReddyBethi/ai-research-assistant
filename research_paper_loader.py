from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login as hf_login
from operator import itemgetter

def c_print(message: str):
    """
    Custom print function for consistent logging.
    """
    prefix = "-> "
    if message.startswith("\n"):
        prefix = "\n-> "
        message = message[1:]
    print(f"{prefix}{message}")

def format_docs(docs):
    """
    Helper function to format a list of Document objects into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)
# ------------------- Load PDF Documents -------------------
all_docs_data = []
data_folder_path = "./data"

for dirpath, dirnames, filenames in os.walk(data_folder_path):
    for file in filenames:
        full_file_path = os.path.join(dirpath, file)
        loader = PyPDFLoader(full_file_path)
        docs = loader.load()
        # The.load() method returns a list of Document objects (one for each page).
        # We use.extend() to add all pages from the current PDF to our main list.
        all_docs_data.extend(docs)
        c_print(f"\n{len(docs)} Pages loaded from: {full_file_path}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

c_print(f"\nSplitting {len(all_docs_data)} pages into chunks...")
splits = text_splitter.split_documents(all_docs_data)
c_print(f"Created {len(splits)} chunks.")
# c_print(f"\nContents of a single chunk:\n{splits[0].page_content}")
# c_print(f"\nMetadata of the chunk:\n{splits[0].metadata}")

# ------------------- Create or Load Vector Store -------------------
c_print("\nInitializing embedding model...")
# 'all-MiniLM-L6-v2' is a popular and efficient model that runs locally
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
)

persist_directory = '.chromaDB'

if os.path.exists(persist_directory):
    c_print(f"Loading existing vector store from '{persist_directory}' folder...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    c_print("Vector store loaded successfully.")
else:
    c_print(f"Creating new vector store in '{persist_directory}' folder...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    c_print("New vector store created and persisted.")

# ------------------- Check if the vector store is working correctly -------------------
# c_print("\n--- Testing the vector store with a similarity search ---")
# test_query = "What is the main contribution of the 'Exergames for telerehabilitation' thesis?"
# retrieved_docs = vectorstore.similarity_search(test_query, k=2) # Retrieve the top 2 most relevant chunks

# c_print(f"Query: '{test_query}'")
# c_print(f"Retrieved {len(retrieved_docs)} documents.")

# for i, doc in enumerate(retrieved_docs):
#     c_print(f"\n--- Document {i+1}: ---")
#     c_print(doc.page_content)
#     c_print("\n--- Metadata: ---")
#     c_print(doc.metadata)

# ------------------- Build the RAG Chain -------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# This prompt is instructs the LLM to answer questions 'only' based on
# the provided context from the papers. This is a key to prevent hallucinations.
template = """
You are an expert research assistant. Your task is to answer questions based on the provided context.
Answer the question based only on the following context.
If the answer cannot be found in the context, write "I am sorry! I could not find the answer in the provided documents.
If you find multiple answers, combine them into a comprehensive response."

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Get Hugging Face token from environment variable
hf_token=os.getenv("HUGGINGFACE_API_KEY")

# Login to Hugging Face. Hugging Faces should automatically show a login prompt if needed.
hf_login(token=hf_token)

# We are using Google's Gemma-2b-it model. It's powerful and runs locally.
c_print("\nInitializing local LLM (Gemma-2b-it)...")
model_id = "google/gemma-2b-it"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, # Same range as float32 but less precision, saves memory
    device_map="auto" # Automatically use GPU if available
)

# Create a text-generation pipeline from the transformers library
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # The maximum number of tokens to generate
    temperature=0.1
)

# Wrap the transformers pipeline in a LangChain object
llm = HuggingFacePipeline(pipeline=pipe)
c_print("LLM initialized successfully.")

rag_chain = (
    {
        "context": itemgetter("context") | RunnableLambda(format_docs),
        "question": itemgetter("question")
    }
    | prompt | llm | StrOutputParser())

query = "What is the purpose of the RehabFork system described in the papers?"
c_print(f"Query: {query}\n")
c_print("Retrieving relevant documents...")
retrieved_docs = retriever.invoke(query)

# # Print the sources (if needed)
# c_print("\nSources: ---")
# for i, doc in enumerate(retrieved_docs):
#     c_print(f"Source {i+1} (from '{doc.metadata.get('source', 'N/A')}', page {doc.metadata.get('page', 'N/A')}):")
#     print(f"\"{doc.page_content[:250]}...\"\n")

# Stream the answer from the LLM
c_print("Generating Answer (Streaming):")
full_response = ""
chain_input = {"context": retrieved_docs, "question": query}

# The.stream() method returns a generator that yields tokens as they are generated.
for chunk in rag_chain.stream(chain_input):
    # Print each chunk as it arrives
    print(chunk, end="", flush=True)
    full_response += chunk

print("\n\n--- Generation Complete ---")

# # RAG chain 1 to answer user's query
# rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# # RAG chain 2 to retrieve and return the source documents
# rag_chain_with_sources = RunnablePassthrough.assign(
#     context=(lambda x: x["question"]) | retriever
# ).assign(
#     answer=(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
#     | prompt | llm | StrOutputParser()
# )

# # ------------------- Check if the RAG chain is working as expected -------------------
# c_print("\n--- Testing the RAG chain ---")
# query = "What is the purpose of the RehabFork system described in the papers?"
# response = rag_chain_with_sources.invoke({"question": query})
# c_print(f"Query: {query}\n")
# c_print("Answer:")
# c_print(response["answer"])
# c_print("\nSources:")
# for i, doc in enumerate(response["context"]):
#     c_print(f"Source {i+1} (from '{doc.metadata.get('source', 'N/A')}', page {doc.metadata.get('page', 'N/A')}):")
#     c_print(f"\"{doc.page_content[:250]}...\"\n")
