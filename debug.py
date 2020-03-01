def FindOffset(tokens_id, answer_id):
    n = len(tokens_id)
    m = len(answer_id)
    if n < m:
        return False
    for i in range(n - m+1):
        if tokens_id[i:i + m] == answer_id:
            return (i, i + m-1)
    return False
