def retrieval_hit(expected: list[str], retrieved: list[str]) -> bool:
    return any(item in retrieved for item in expected)