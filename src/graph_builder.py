from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path
import pickle
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class Graph_Builder:

    def __init__(self,chunks=None):
        self.chunks=chunks

    def graph_vector(self, threshold=0.5):

        GRAPH_PATH = "Graph_db/graph.pkl"
        VECTOR_DB_PATH = "Graph_db/chroma"

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        os.makedirs("Graph_db", exist_ok=True)

        # Load existing databases
        if os.path.exists(GRAPH_PATH):

            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )

            with open(GRAPH_PATH, "rb") as file:
                graph = pickle.load(file)

            print("Graph vector store loaded")

            return vector_store, graph

        # Create vector database
        vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        # Create graph
        graph = nx.Graph()

        texts = [chunk.page_content for chunk in self.chunks]
        vectors = embeddings.embed_documents(texts)

        # Add nodes
        for i, chunk in enumerate(self.chunks):
            graph.add_node(
                i,
                text=chunk.page_content,
                metadata=chunk.metadata
            )

        # Add similarity edges
        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):

                score = cosine_similarity(
                    [vectors[i]],
                    [vectors[j]]
                )[0][0]

                if score >= threshold:
                    graph.add_edge(
                        i,
                        j,
                        weight=float(score)
                    )

        # Save graph locally
        with open(GRAPH_PATH, "wb") as file:
            pickle.dump(graph, file)

        print("Graph vector store created")

        return vector_store, graph