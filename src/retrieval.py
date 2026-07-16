class Retrieve:
    def __init__(self):
        pass
    def top_K(self,db,question,topk=5):
        retriever =db.as_retriever(search_kwargs={"k":topk})
        retrieved_docs = retriever.invoke(question)

        context ="\n\n".join(i.page_content for i in retrieved_docs)
        return context
    def MMR(self,db,question,topk=5):
        retrieved_docs = db.max_marginal_relevance_search(
            query=question, k=topk, fetch_k=20, lambda_mult=0.2
        )
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context