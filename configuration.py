from src.chunking import Chunking

class Chunk_selector:

    def __init__(self,method,doc):
        self.method=method
        self.doc=doc

    def split_documents(self):

        ch=Chunking(self.doc)

        if self.method.lower()=="overlap":

            chunks=ch.overlap_chunker()

            return chunks
        
        elif self.method.lower()=="semantic":

            chunks=ch.semantic_chunker()

            return chunks
        
        else:

            raise ValueError("Enter only Overlap or Semantic")
