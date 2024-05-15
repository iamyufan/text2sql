# Function to load the texts and gt queries
def load_data(file_nl, file_sql=None):
    with open(file_nl, "r", encoding="utf-8") as f_nl:
        texts = f_nl.readlines()
    queries = None
    if file_sql:
        with open(file_sql, "r", encoding="utf-8") as f_sql:
            queries = f_sql.readlines()
    return texts, queries

