import sys

import onnx


def main(path):
    print("Inspecting:", path)
    m = onnx.load(path)
    print("Graph outputs:")
    for o in m.graph.output:
        print(" -", o.name)

    print('\nSearching nodes that emit "logits" in outputs:')
    found = 0
    for i, n in enumerate(m.graph.node):
        outs = list(n.output)
        if any("logit" in s.lower() for s in outs):
            found += 1
            print(f"Node #{i}: name={n.name!r} op_type={n.op_type!r} outputs={outs}")

    print("\nTotal nodes matching logits substring:", found)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dump_graph_outputs.py <model.onnx>")
        sys.exit(2)
    main(sys.argv[1])
