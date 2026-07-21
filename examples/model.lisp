(let-dim in 784)
(let-dim d 128)

(define encoder (mlp in (256 256) d))
(define model (chain
  (mlp in (256) d)
  (layernorm d)))
