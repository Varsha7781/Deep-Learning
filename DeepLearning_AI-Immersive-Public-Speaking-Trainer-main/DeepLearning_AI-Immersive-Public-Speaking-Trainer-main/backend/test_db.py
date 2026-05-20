from database import SessionLocal
from models import Analysis, Recording

db = SessionLocal()

print("Analysis rows:", db.query(Analysis).count())
print("Recording rows:", db.query(Recording).count())