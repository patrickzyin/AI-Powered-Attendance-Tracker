import sqlite3
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM persons')
print(cursor.fetchall())
conn.close()