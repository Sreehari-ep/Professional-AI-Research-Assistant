import sqlite3
from datetime import datetime
from utils.paths import DATABASE_FOLDER
DB_PATH=DATABASE_FOLDER/"research_assistant.db"

def connect(): return sqlite3.connect(DB_PATH)

def create_tables():
    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY AUTOINCREMENT,filename TEXT UNIQUE,uploaded_at TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT,content TEXT,category TEXT,linked_document TEXT,tags TEXT,created_at TEXT)")

def save_document(filename):
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO documents(filename,uploaded_at) VALUES(?,?)",(filename,datetime.now().isoformat(timespec="seconds")))

def get_documents():
    with connect() as con:
        return con.execute("SELECT id,filename,uploaded_at FROM documents ORDER BY id DESC").fetchall()

def delete_document(document_id):
    with connect() as con: con.execute("DELETE FROM documents WHERE id=?",(document_id,))

def count_documents():
    with connect() as con: return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

def create_note(title,content,category="General",linked_document="",tags=""):
    with connect() as con:
        con.execute("INSERT INTO notes(title,content,category,linked_document,tags,created_at) VALUES(?,?,?,?,?,?)",
                    (title,content,category,linked_document,tags,datetime.now().isoformat(timespec="seconds")))

def get_notes(search=""):
    with connect() as con:
        if search:
            q=f"%{search}%"
            rows=con.execute("SELECT id,title,content,category,linked_document,tags,created_at FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY id DESC",(q,q)).fetchall()
        else:
            rows=con.execute("SELECT id,title,content,category,linked_document,tags,created_at FROM notes ORDER BY id DESC").fetchall()
    keys=["id","title","content","category","linked_document","tags","created_at"]
    return [dict(zip(keys,row)) for row in rows]

def update_note(*args,**kwargs): return None

def delete_note(note_id):
    with connect() as con: con.execute("DELETE FROM notes WHERE id=?",(note_id,))
