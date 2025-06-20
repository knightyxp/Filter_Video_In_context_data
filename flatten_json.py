import json
import sys
import os

def flatten_json(input_file, output_file):
    """
    Flatten a JSON file containing nested lists of dictionaries into a single list of dictionaries.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten nested lists
    flattened = []
    for item in data:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write flattened data to new JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(flattened, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    flatten_json(input_path, output_path)
    print(f"Flattened JSON saved to: {output_path}")
