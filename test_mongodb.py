"""Optional MongoDB connectivity check. Does not print credentials."""
import logging
from database import Database

logging.basicConfig(level=logging.INFO)
db = Database()
if db.connected:
    print("MongoDB connection succeeded.")
else:
    print("MongoDB connection failed or is unavailable. Check .env and network access.")
db.close()
