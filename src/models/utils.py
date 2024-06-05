def tokens_to_feature_map(tokens): 
    B, N, D = tokens.shape 
    H = W = int(N ** 0.5)
    assert H * W == N

    tokens = tokens.view(B, H, W, D)
    tokens = tokens.permute(0, 3, 1, 2)

    return tokens