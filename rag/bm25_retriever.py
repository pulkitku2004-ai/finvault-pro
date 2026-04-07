#bm25 == best matching 25, while vector search understands the vibe(understands meaning) of the code,
#bm25 finds the exact keywords used.
#Logic - 3 main factors : 1) Term frequency: in which chunk the word has repeated more times, thats more relevant,
# 2) Inverse Document Frequency: if the word appears in every document, then its useless, but if in one oage, then its imp,
# 3) Document Length Normalization: If a 100-page book mentions "Profit" 50 times, it's not as impressive as a 1-page summary mentioning "Profit" 10 times.

from rank_bm25 import BM25Okapi

def bm25_search(query, documents, top_k=5):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(query.split())

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]