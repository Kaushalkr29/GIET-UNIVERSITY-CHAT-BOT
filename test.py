from src.document_loader import Document_Loader
from src.chunking import Chunking
from src.embeddings import Embedding
from src.generation import Generation
from src.graph_builder import Graph_Builder
from src.retrieval import Retrieve


d=Document_Loader()
doc=d.file_loader()
c=Chunking(doc)
chunks=c.semantic_chunker()
g=Graph_Builder(chunks)
vector_db=g.graph_vector()

