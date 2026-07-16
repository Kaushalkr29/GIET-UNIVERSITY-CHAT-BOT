from langchain_huggingface import HuggingFaceEmbeddings

class Embedding:

    def __init__(self):
        pass

    def embedding_model(self):
        model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return model