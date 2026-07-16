from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

class Ingest:

    def __init__(self,chunks):
        self.chunks=chunks

    def vectorizer(self):

        chroma_dir="Chroma_db"

        if Path(chroma_dir).exists():
            choice = input(
                "Vector DB already exists.\n"
                "Rebuild database? (y/n): "
            )

            if choice.lower() == "y":
                shutil.rmtree(chroma_dir)
                print("Old ChromaDB deleted.")
                print("Creating New Chroma DB")
                vector_db=Chroma.from_documents(documents=self.chunks,
                                                persist_directory=chroma_dir,
                                                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
                print("Embedding Model Loaded.")
                print(f"chunks Loaded in Vector DB {len(self.chunks)}")
            else:
                print("Using existing ChromaDB.")
                vector_db=Chroma(persist_directory=chroma_dir,
                                 embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
            
        else:
            print("Creating New Chroma DB")
            vector_db=Chroma.from_documents(documents=self.chunks,
                                            persist_directory=chroma_dir,
                                            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
            print("Embedding Model Loaded.")
            print(f"self.chunks Loaded in Vector DB {len(self.chunks)}")
        return vector_db

        


