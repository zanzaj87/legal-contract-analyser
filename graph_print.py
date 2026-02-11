#'from graph import compile_graph
from graph_with_tools import compile_graph


compiled = compile_graph()
png_bytes = compiled.get_graph().draw_mermaid_png()

with open("graph_diagram.png", "wb") as f:
    f.write(png_bytes)

print("Saved to graph_diagram.png")