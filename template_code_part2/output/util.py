# Add your import statements here
import json





# Utility helper for loading only the document text from the Cranfield JSON file.
def get_docs(file_path):
    # Read the dataset file as a list of JSON objects.
    with open(file_path, "r") as f:
        data = json.load(f)

    documents = []

    for entry in data:
        # Keep just the document body so the downstream pipeline sees raw text.
        content =  entry["body"]
        documents.append(content)
    return documents
