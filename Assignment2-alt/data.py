import json

def load_data(file_path):
    d = []
    with open(file_path, 'r') as file:
        for l in file:
            row = json.loads(l)
            d.append(row)
    return d