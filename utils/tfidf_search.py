from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def tfidf_search(query,chunks,metadata=None,top_k=8):
    if not chunks: return []
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
    matrix=vectorizer.fit_transform(chunks+[query]); scores=cosine_similarity(matrix[-1],matrix[:-1]).flatten()
    out=[]
    for rank,i in enumerate(scores.argsort()[::-1][:top_k],1):
        item={"rank":rank,"chunk":chunks[i],"score":float(scores[i]),"score_percentage":round(float(scores[i])*100,2)}
        if metadata: item.update(metadata[i])
        out.append(item)
    return out
