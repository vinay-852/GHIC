import sqlite3

# ---- CONFIG ----
DB_PATH = "app_data.db"
TABLE_NAME = "query_history"
PRIMARY_KEY = "id"   # column name to identify each row
# -----------------

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Fetch all rows first
cursor.execute(f"SELECT {PRIMARY_KEY}, * FROM {TABLE_NAME}")
rows = cursor.fetchall()

print(f"Total rows found: {len(rows)}")
print("Press ENTER to delete a row, or type anything to skip.\n")

for row in rows:
    row_id = row[0]
    print(f"Row ID = {row_id} | Data = {row}")

    user_input = input("Delete this row? (ENTER = delete / anything else = skip): ")

    if user_input.strip() == "":
        cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE {PRIMARY_KEY} = ?", (row_id,))
        conn.commit()
        print("✔ Deleted\n")
    else:
        print("⏭ Skipped\n")

print("Done!")
conn.close()
