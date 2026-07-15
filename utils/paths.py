from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
UPLOAD_FOLDER=BASE_DIR/"uploads"
REPORTS_FOLDER=BASE_DIR/"reports"
DATABASE_FOLDER=BASE_DIR/"database"
VECTOR_FOLDER=BASE_DIR/"vector_db"
for folder in (UPLOAD_FOLDER,REPORTS_FOLDER,DATABASE_FOLDER,VECTOR_FOLDER):
    folder.mkdir(parents=True,exist_ok=True)
