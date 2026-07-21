(let-dim
  (d 128) (heads 8) (layers 4))

(define attn_block (residual (chain (layernorm d) (attention d heads causal))))
(define ffn_block (residual (chain (layernorm d) (mlp d [4d] d))))

(define encoder (repeat layers (chain attn_block ffn_block)))
