"""
One-time script: remove null bytes from Python files in factory/.
Run from project root:  python fix_null_bytes.py
"""
import os

BASE = os.path.join(os.path.dirname(__file__), "factory")
for name in ("urls.py", "views.py", "__init__.py", "models.py", "admin.py"):
    path = os.path.join(BASE, name)
    if not os.path.isfile(path):
        continue
    with open(path, "rb") as f:
        data = f.read()
    if b"\x00" in data:
        clean = data.replace(b"\x00", b"")
        with open(path, "wb") as f:
            f.write(clean)
        print("Fixed:", path)
    else:
        print("OK (no nulls):", path)
print("Done.")
