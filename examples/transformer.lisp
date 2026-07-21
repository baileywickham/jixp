(let-dim d 128)
(let-dim heads 8)

(define block (chain
  (residual (chain (layernorm d) (attention d heads)))
  (residual (chain (layernorm d) (mlp d [4d] d)))))
