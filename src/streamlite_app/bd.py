import sqlite3

def view_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions")
    rows = c.fetchall()
    for row in rows:
        print(row)
    conn.close()

view_predictions()
