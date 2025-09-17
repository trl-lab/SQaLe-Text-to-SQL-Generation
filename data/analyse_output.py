import json

def calculate_empty_tables_share(json_input):
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError:
            try:
                with open(json_input, 'r') as file:
                    data = json.load(file)
            except (IOError, json.JSONDecodeError):
                raise ValueError("Input must be a valid JSON string or file path.")
    else:
        data = json_input

    files = data.get('files', [])

    if not files:
        return 0.0

    empty_count = 0
    for file_entry in files:
        tables = file_entry.get('tables', [])
        if not tables:
            empty_count += 1

    total_files = len(files)
    share = (empty_count / total_files) * 100 if total_files > 0 else 0.0

    return share

# Load JSON data from output.json
with open('data/output.json', 'r') as f:
    json_data = json.load(f)

share = calculate_empty_tables_share(json_data)
print(f"Share of files with empty tables lists: {share}%")
