import type { ModelNode } from "./types";

/** Top-to-bottom data-flow view: blocks on a spine, residuals as skip arcs
 *  into a ⊕ merge, repeat as a dashed ×N container. */

const fmt = (n: number): string =>
  n >= 1e9 ? (n / 1e9).toFixed(2) + "B" :
  n >= 1e6 ? (n / 1e6).toFixed(2) + "M" :
  n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);

function spliceChain(n: ModelNode): ModelNode[] | undefined {
  let c = "children" in n ? n.children : undefined;
  while (c && c.length === 1 && c[0].kind === "chain")
    c = c[0].children;
  return c;
}

function label(n: ModelNode): string {
  switch (n.kind) {
    case "mlp": return `mlp ${n.dims.join(" → ")} · ${n.activation}${n.bias ? " · bias" : ""}`;
    case "layernorm": return `layernorm d=${n.out}`;
    case "rmsnorm": return `rmsnorm d=${n.out}${n.scale ? "" : " · no scale"}`;
    case "embedding": return `embedding vocab=${n.vocab} → d=${n.out}`;
    case "attention": return `attention d=${n.out} · heads=${n.heads}${n.causal ? " · causal" : ""}${n.rope ? " · rope" : ""}`;
    default: return n.kind;
  }
}

function Connector() {
  return <div className="mx-auto h-3 w-0.5 bg-[var(--muted)] opacity-50" />;
}

function BlockBox({ n, denom, onJump }: { n: ModelNode; denom: number; onJump: (name: string) => void }) {
  const pct = denom ? (100 * n.params) / denom : 0;
  const isRef = n.kind === "ref";
  return (
    <div
      className="mx-auto w-full max-w-[420px] rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5"
      style={{ borderLeft: `4px solid var(--k-${n.kind})` }}
      title={`${n.params.toLocaleString()} params`}
    >
      <div className="flex items-baseline gap-2">
        {isRef ? (
          <span
            className="cursor-pointer text-[13px] font-semibold text-[var(--accent)] underline decoration-dotted underline-offset-[3px]"
            onClick={() => n.kind === "ref" && onJump(n.ref)}
          >
            {n.kind === "ref" ? n.ref : ""}
          </span>
        ) : (
          <span className="text-[13px]">{label(n)}</span>
        )}
        <span className="ml-auto text-xs tabular-nums text-[var(--muted)]">{fmt(n.params)}</span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-sm bg-[var(--grid)]">
        <div
          className="h-full min-w-[2px] rounded-sm bg-[var(--accent)]"
          style={{ width: `${Math.max(pct, 0.5)}%` }}
        />
      </div>
    </div>
  );
}

export function FlowNode({ n, denom, onJump }: { n: ModelNode; denom: number; onJump: (name: string) => void }) {
  const kids = spliceChain(n);

  if (n.kind === "residual" && kids?.length) {
    return (
      <div className="relative mx-auto w-full max-w-[488px] pl-7 pr-7">
        {/* skip lane: down the left, around the body, into the merge node */}
        <div
          className="absolute bottom-[11px] left-1.5 top-[-12px] w-4 rounded-l-xl border-2 border-r-0 opacity-70"
          style={{ borderColor: "var(--k-residual)" }}
        />
        <div className="flex flex-col">
          {kids.map((c, i) => (
            <div key={i}>
              {i > 0 && <Connector />}
              <FlowNode n={c} denom={denom} onJump={onJump} />
            </div>
          ))}
        </div>
        <Connector />
        <div
          className="mx-auto flex h-[22px] w-[22px] items-center justify-center rounded-full border-2 bg-[var(--surface)] text-[13px] font-bold"
          style={{ borderColor: "var(--k-residual)", color: "var(--k-residual)" }}
          title="x + body(x)"
        >
          +
        </div>
      </div>
    );
  }

  if (n.kind === "repeat" && kids?.length) {
    return (
      <div
        className="mx-auto w-full max-w-[520px] rounded-xl border-2 border-dashed p-2"
        style={{ borderColor: "color-mix(in srgb, var(--k-repeat) 55%, transparent)" }}
      >
        <div className="mb-1 flex items-baseline px-1 text-[12px] text-[var(--ink-2)]">
          <span className="font-semibold" style={{ color: "var(--k-repeat)" }}>
            repeat × {n.count}
          </span>
          <span className="ml-auto text-xs tabular-nums text-[var(--muted)]">{fmt(n.params)}</span>
        </div>
        <div className="flex flex-col">
          {kids.map((c, i) => (
            <div key={i}>
              {i > 0 && <Connector />}
              <FlowNode n={c} denom={denom} onJump={onJump} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (kids?.length) {
    // plain chain: stack with connectors
    return (
      <div className="flex flex-col">
        {kids.map((c, i) => (
          <div key={i}>
            {i > 0 && <Connector />}
            <FlowNode n={c} denom={denom} onJump={onJump} />
          </div>
        ))}
      </div>
    );
  }

  return <BlockBox n={n} denom={denom} onJump={onJump} />;
}

export default function FlowView({ node, denom, onJump }: { node: ModelNode; denom: number; onJump: (name: string) => void }) {
  return (
    <div className="py-2">
      <div className="mx-auto mb-1 w-fit rounded border border-[var(--border)] px-2 py-0.5 text-xs text-[var(--muted)]">
        x : {String(node.in)}
      </div>
      <Connector />
      <FlowNode n={node} denom={denom} onJump={onJump} />
      <Connector />
      <div className="mx-auto w-fit rounded border border-[var(--border)] px-2 py-0.5 text-xs text-[var(--muted)]">
        out : {String(node.out)}
      </div>
    </div>
  );
}
