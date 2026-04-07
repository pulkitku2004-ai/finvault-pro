from ingestion.pdf_parser import parse_pdf
from graph.graph_builder import GraphBuilder


docs = parse_pdf("data/hdfc_q3.pdf")

builder = GraphBuilder()

for doc in docs:

    # Check if doc is a string or an object
    if hasattr(doc, 'page_content'):
        text = doc.page_content
    else:
        text = doc # It's already a string!

    builder.create_graph(text)

builder.close()

print("Graph successfully created.")