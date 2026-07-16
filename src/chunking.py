from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import Embedding

class Chunking:
    def __init__(self,documents):
        self.documents=documents
        print(f"Original documents: {len(self.documents)}")

    def overlap_chunker(self):

        splitter=RecursiveCharacterTextSplitter(chunk_size=650,
                                                chunk_overlap=115)
        overlap_chunks=splitter.split_documents(self.documents)

        print(f"Overlapping Chunks: {len(overlap_chunks)}")

        for i, chunk in enumerate(overlap_chunks,start=1):

            source = chunk.metadata.get("source", "unknown")
            file_type = chunk.metadata.get("file_type", "unknown")

            source_name = source.replace(".", "_")

            chunk.metadata["chunk_id"] = (
                f"{source_name}_chunk_{i:04d}"
            )

            chunk.metadata["chunk_index"] = i


        return overlap_chunks
    
    def semantic_chunker(self):
        m=Embedding()
        splitter=SemanticChunker(embeddings=m.embedding_model(),breakpoint_threshold_type="percentile")
        sem_chunks=splitter.split_documents(self.documents)

        print(f"Semantic Chunks: {len(sem_chunks)}")

        for i, chunk in enumerate(sem_chunks,start=1):

            source = chunk.metadata.get("source", "unknown")
            file_type = chunk.metadata.get("file_type", "unknown")

            source_name = source.replace(".", "_")

            chunk.metadata["chunk_id"] = (
                f"{source_name}_chunk_{i:04d}"
            )

            chunk.metadata["chunk_index"] = i

        return sem_chunks
    
