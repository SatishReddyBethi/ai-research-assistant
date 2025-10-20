from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

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
        print(f"\n-> {len(docs)} Pages loaded from: {full_file_path}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

print(f"\nSplitting {len(all_docs_data)} pages into chunks...")
splits = text_splitter.split_documents(all_docs_data)
print(f"-> Created {len(splits)} chunks.")
# print(f"\n-> Contents of a single chunk:\n{splits[0].page_content}")
# print(f"\n-> Metadata of the chunk:\n{splits[0].metadata}")

# ------------------- Create or Load Vector Store -------------------
print("Initializing embedding model...")
# 'all-MiniLM-L6-v2' is a popular and efficient model that runs locally
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
)

persist_directory = '.chromaDB'

if os.path.exists(persist_directory):
    print(f"Loading existing vector store from '{persist_directory}'...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    print("Vector store loaded successfully.")
else:
    print(f"Creating new vector store in '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("New vector store created and persisted.")

# ------------------- Check if the vector store is working correctly -------------------
# print("\n--- Testing the vector store with a similarity search ---")
# test_query = "What is the main contribution of the 'Exergames for telerehabilitation' thesis?"
# retrieved_docs = vectorstore.similarity_search(test_query, k=2) # Retrieve the top 2 most relevant chunks

# print(f"Query: '{test_query}'")
# print(f"Retrieved {len(retrieved_docs)} documents.")

# for i, doc in enumerate(retrieved_docs):
#     print(f"\n--- Document {i+1}: ---")
#     print(doc.page_content)
#     print("\n--- Metadata: ---")
#     print(doc.metadata)

