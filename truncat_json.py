import json

F = "vie_score_obj_removal.json"
with open(F, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果你的 JSON 结构是一个数组，直接用 data = data[:-97]；否则假设是 {"results": [...]}
if isinstance(data, dict) and "results" in data:
    orig = len(data["results"])
    data["results"] = data["results"][:-97]
    print(f"Removed last 97 entries (from {orig} to {len(data['results'])})")
else:
    orig = len(data)
    data = data[:-97]
    print(f"Removed last 97 entries (from {orig} to {len(data)})")

with open(F, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
