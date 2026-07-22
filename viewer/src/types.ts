/** Mirror of compiler.model_summary()'s JSON schema. */

export type Dim = number | "tokens";

interface BaseNode {
  in: Dim;
  out: Dim;
  params: number;
}

export interface MlpNode extends BaseNode {
  kind: "mlp";
  dims: number[];
  activation: string;
  bias: boolean;
}

export interface LayernormNode extends BaseNode {
  kind: "layernorm";
}

export interface EmbeddingNode extends BaseNode {
  kind: "embedding";
  vocab: number;
}

export interface AttentionNode extends BaseNode {
  kind: "attention";
  heads: number;
  causal: boolean;
}

export interface ChainNode extends BaseNode {
  kind: "chain";
  children: ModelNode[];
}

export interface ResidualNode extends BaseNode {
  kind: "residual";
  children: ModelNode[];
}

export interface RepeatNode extends BaseNode {
  kind: "repeat";
  count: number;
  children: ModelNode[];
}

export interface RefNode extends BaseNode {
  kind: "ref";
  ref: string;
}

export type ModelNode =
  | MlpNode
  | LayernormNode
  | EmbeddingNode
  | AttentionNode
  | ChainNode
  | ResidualNode
  | RepeatNode
  | RefNode;

export interface Define {
  name: string;
  mangled: string;
  node: ModelNode;
}

export interface Summary {
  defines: Define[];
  total_params: number;
}
