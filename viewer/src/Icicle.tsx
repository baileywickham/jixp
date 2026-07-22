import type { ModelNode } from "./types";

/** Icicle param map: width = parameter share, rows = structure depth.
 *  Inside a repeat the single body copy is scaled by count so widths stay
 *  proportional to the real budget. */

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

function segmentLabel(n: ModelNode): string {
  if (n.kind === "ref") return n.ref;
  if (n.kind === "repeat") return `repeat × ${n.count}`;
  return n.kind;
}

function Segment({ n, onJump }: { n: ModelNode; onJump: (name: string) => void }) {
  const isRef = n.kind === "ref";
  return (
    <div
      className={`h-[30px] overflow-hidden whitespace-nowrap rounded-[3px] px-1.5 text-[11.5px] leading-[30px] text-[var(--ink)] ${isRef ? "cursor-pointer underline decoration-dotted underline-offset-[3px]" : ""}`}
      style={{ background: `color-mix(in srgb, var(--k-${n.kind}) 22%, var(--surface))`,
               borderLeft: `3px solid var(--k-${n.kind})` }}
      title={`${segmentLabel(n)} — ${n.params.toLocaleString()} params`}
      onClick={isRef && n.kind === "ref" ? () => onJump(n.ref) : undefined}
    >
      {segmentLabel(n)} <span className="text-[var(--muted)]">{fmt(n.params)}</span>
    </div>
  );
}

export default function IcicleView({ node, onJump }: { node: ModelNode; onJump: (name: string) => void }) {
  const kids = spliceChain(node);
  const scale = node.kind === "repeat" ? node.count : 1;
  return (
    <div className="flex min-w-0 flex-col gap-[3px]">
      <Segment n={node} onJump={onJump} />
      {kids?.length ? (
        <div className="flex min-w-0 flex-row gap-[3px]">
          {kids.map((c, i) => (
            <div
              key={i}
              className="min-w-0"
              style={{ flexGrow: Math.max(c.params * scale, node.params * 0.03), flexBasis: 0 }}
            >
              <IcicleView node={c} onJump={onJump} />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
