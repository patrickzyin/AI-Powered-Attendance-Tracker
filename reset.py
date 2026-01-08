#!/usr/bin/env python3
"""
Clean Reset Script
Deletes all data and starts fresh
"""

import os
import shutil

print("=" * 60)
print("🧹 CLEAN RESET - Deleting All Data")
print("=" * 60)

# Files and folders to delete
items_to_delete = [
    'attendance.db',
    'secret.key',
    'registered_faces',
    'uploads',
    '__pycache__',
    'flask_session'
]

deleted_count = 0

for item in items_to_delete:
    try:
        if os.path.isfile(item):
            os.remove(item)
            print(f"✓ Deleted file: {item}")
            deleted_count += 1
        elif os.path.isdir(item):
            shutil.rmtree(item)
            print(f"✓ Deleted folder: {item}")
            deleted_count += 1
    except FileNotFoundError:
        print(f"⊘ Not found: {item}")
    except Exception as e:
        print(f"✗ Error deleting {item}: {e}")

print("=" * 60)
if deleted_count > 0:
    print(f"✅ Cleaned {deleted_count} item(s)")
    print("\n💡 All data deleted! Run 'python app.py' to start fresh")
else:
    print("✅ Already clean - no data to delete")
print("=" * 60)