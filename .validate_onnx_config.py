import json

p = "c:/Workspace/GitHub/Flouds.Py/app/config/onnx_config.json"
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
print(
    "decoder_input_name:",
    repr(data.get("instructor-xl", {}).get("inputnames", {}).get("decoder_input_name")),
)
