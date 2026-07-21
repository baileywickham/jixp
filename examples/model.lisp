(define encoder (mlp 784 (256 256) 128))
(define model (chain
  (mlp 784 (256) 128)
  (layernorm 128)))
