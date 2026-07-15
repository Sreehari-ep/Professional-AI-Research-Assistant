import re
from collections import Counter
STOP={"the","and","for","with","that","this","from","were","have","using","used","study","research","paper","results","method","methods","data","analysis"}

def chunk_text(text,chunk_size=5,overlap=1):
    sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if len(s.strip())>20]
    if not sentences: return [text[:2000]]
    step=max(chunk_size-overlap,1)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0,len(sentences),step)]

def extract_keywords(text,top_n=15):
    words=re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b",text.lower())
    return [w for w,_ in Counter(w for w in words if w not in STOP).most_common(top_n)]
