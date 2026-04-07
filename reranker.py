from sentence_transformers import CrossEncoder

# Cross encoder reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, docs, top_k=3):

    pairs = [[query, doc] for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]

#this used a cross-encoder reranking model, reliable.
#no ONNX proble.