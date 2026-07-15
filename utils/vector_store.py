import pickle,faiss,numpy as np
from utils.paths import VECTOR_FOLDER
INDEX=VECTOR_FOLDER/"faiss.index"; CHUNKS=VECTOR_FOLDER/"chunks.pkl"; META=VECTOR_FOLDER/"metadata.pkl"
def _load(path,default):
    if not path.exists(): return default
    with open(path,"rb") as f: return pickle.load(f)
def _save(path,value):
    with open(path,"wb") as f: pickle.dump(value,f)
def load_index(): return faiss.read_index(str(INDEX)) if INDEX.exists() else None
def add_documents(chunks,embeddings,filename):
    embeddings=np.asarray(embeddings,dtype="float32")
    index=load_index() or faiss.IndexFlatIP(embeddings.shape[1])
    old_chunks=_load(CHUNKS,[]); old_meta=_load(META,[]); start=len(old_chunks)
    index.add(embeddings); old_chunks.extend(chunks)
    old_meta.extend({"filename":filename,"chunk_id":start+i} for i in range(len(chunks)))
    faiss.write_index(index,str(INDEX)); _save(CHUNKS,old_chunks); _save(META,old_meta)
def search_documents(query_embedding,top_k=8):
    index=load_index(); chunks=_load(CHUNKS,[]); meta=_load(META,[])
    if index is None: return []
    scores,ids=index.search(np.asarray(query_embedding,dtype="float32"),min(top_k,index.ntotal))
    out=[]
    for rank,(i,s) in enumerate(zip(ids[0],scores[0]),1):
        if i<0: continue
        out.append({"rank":rank,"filename":meta[i]["filename"],"chunk_id":meta[i]["chunk_id"],"chunk":chunks[i],"score":float(s),"score_percentage":round(max(float(s),0)*100,2)})
    return out
def vector_store_info():
    index=load_index()
    return {"vectors":int(index.ntotal) if index else 0,"dimension":int(index.d) if index else 0,"chunks":len(_load(CHUNKS,[])),"metadata":len(_load(META,[]))}
