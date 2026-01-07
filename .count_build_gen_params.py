import ast
from pathlib import Path

p = Path(__file__).parent / "app" / "services" / "prompt_service.py"
s = p.read_text(encoding="utf-8")
mod = ast.parse(s)
count = None
for node in mod.body:
    if isinstance(node, ast.ClassDef) and node.name == "PromptProcessor":
        for n in node.body:
            if isinstance(n, ast.FunctionDef) and n.name == "_build_generation_params":
                count = n.end_lineno - n.lineno + 1
                print(count)
                break
        break
if count is None:
    print("function not found")
