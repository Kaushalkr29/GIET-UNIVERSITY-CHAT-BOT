from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader

class Document_Loader:

    def __init__(self):
        pass

    def file_loader(self):

        documents=[]
        documents+=DirectoryLoader("data",glob="**/*.pdf",loader_cls=PyPDFLoader).load()
        documents+=DirectoryLoader("data",glob="**/*.docx",loader_cls=Docx2txtLoader).load()
        documents+=DirectoryLoader("data",glob="**/*.txt",loader_cls=TextLoader).load()
        documents+=DirectoryLoader("data",glob="**/*.json",loader_cls=JSONLoader, loader_kwargs={"jq_schema":".","text_content":False}).load()
        print("All Documents Loaded")

        sources=sorted(set(doc.metadata.get("source","Unknown")
                           for doc in documents))
        
        print("Loaded Sources: ")
        for source in sources:
            print(f"-{source}")

        return documents
        
