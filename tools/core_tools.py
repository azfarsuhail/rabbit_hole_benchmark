from langchain_core.tools import tool

@tool
def read_local_document(file_name: str) -> str:
    """Reads a local text document. Useful for extracting financial reports, revenue data, or corporate policies."""
    # Mock implementation for the benchmark
    if "Q3" in file_name or "revenue" in file_name.lower():
        return "The total revenue for Q3 is $4.2 million."
    return "Document not found or empty."

@tool
def calculate_percentage(part: float, whole: float) -> str:
    """Calculates the percentage of a part relative to a whole. Useful for financial math."""
    if whole == 0:
        return "Error: Division by zero."
    return f"The result is {(part / whole) * 100}%"

# We group them so we can easily import them later
core_tools_list = [read_local_document, calculate_percentage]