from langchain_community.document_loaders import PyPDFLoader
import os

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

