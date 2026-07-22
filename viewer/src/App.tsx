import { useRef } from "react";
import type { Define, ModelNode, Summary } from "./types";
import sample from "./sample.json";

// __DATA__ is injected by the jixp compiler; under `npm run dev` the
// placeholder doesn't parse, so the bundled sample model renders instead.
function loadData(): Summary {
  try {
    return JSON.parse(document.getElementById("model-data")!.textContent!);
  } catch {
    return sample as Summary;
  }
}

const fmt = (n: number): string =>
  n >= 1e9 ? (n / 1e9).toFixed(2) + "B" :
  n >= 1e6 ? (n / 1e6).toFixed(2) + "M" :
  n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);

/** Implicit-chain wrappers (an only-child chain) splice into their parent. */
function effectiveChildren(n: ModelNode): ModelNode[] | undefined {
  let c = "children" in n ? n.children : undefined;
  while (c && c.length === 1 && c[0].kind === "chain")
    c = c[0].children;
  return c;
}

function jumpToDefine(name: string): void {
  const card = document.getElementById(`def-${name}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "start" });
  card.classList.remove("flash");
  void card.offsetWidth;
  card.classList.add("flash");
}

const dims = "text-[var(--muted)]";

function RefLink({ name }: { name: string }) {
  return (
    <span
      className="cursor-pointer font-semibold text-[var(--accent)] underline decoration-dotted underline-offset-[3px]"
      onClick={() => jumpToDefine(name)}
    >
      {name}
    </span>
  );
}

function Detail({ n }: { n: ModelNode }) {
  switch (n.kind) {
    case "mlp":
      return (
        <>
          <span className={dims}>{n.dims.join(" → ")}</span> · {n.activation}
          {n.bias ? " · bias" : ""}
        </>
      );
    case "layernorm":
      return <span className={dims}>d={n.out}</span>;
    case "embedding":
      return <span className={dims}>vocab={n.vocab} → d={n.out}</span>;
    case "attention":
      return (
        <>
          <span className={dims}>d={n.out} · heads={n.heads}</span>
          {n.causal ? " · causal" : ""}
        </>
      );
    case "chain":
      return <span className={dims}>{n.in} → {n.out} · {n.children.length} stages</span>;
    case "residual":
      return <span className={dims}>d={n.out}</span>;
    case "repeat":
      return (
        <>
          × {n.count} <span className={dims}>(independent params)</span>
        </>
      );
    case "ref":
      return <span className={dims}>(fresh params)</span>;
  }
}

function Row({ n, denom, leaf = false }: { n: ModelNode; denom: number; leaf?: boolean }) {
  const pct = denom ? (100 * n.params) / denom : 0;
  return (
    <div
      className="grid grid-cols-[14px_10px_84px_minmax(120px,1fr)_minmax(90px,220px)_58px_92px]
                 items-center gap-2.5 rounded-md px-1.5 py-1
                 hover:bg-[color-mix(in_srgb,var(--ink)_5%,transparent)]"
      title={`${n.params.toLocaleString()} params`}
    >
      <span
        className={`caret w-3.5 text-center text-[10px] text-[var(--muted)] transition-transform duration-100 ${leaf ? "invisible" : ""}`}
      >
        ▶
      </span>
      <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: `var(--k-${n.kind})` }} />
      <span className="overflow-hidden text-ellipsis whitespace-nowrap text-[12.5px] text-[var(--ink-2)]">
        {n.kind === "ref" ? <RefLink name={n.ref} /> : n.kind}
      </span>
      <span className="overflow-hidden text-ellipsis whitespace-nowrap">
        <Detail n={n} />
      </span>
      <span className="h-1.5 overflow-hidden rounded-[3px] bg-[var(--grid)]">
        <i
          className="block h-full min-w-[2px] rounded-[3px] bg-[var(--accent)]"
          style={{ width: `${Math.max(pct, 0.75)}%` }}
        />
      </span>
      <span className="text-right text-xs tabular-nums text-[var(--muted)]">{pct.toFixed(1)}%</span>
      <span className="text-right text-[12.5px] tabular-nums text-[var(--ink-2)]">{fmt(n.params)}</span>
    </div>
  );
}

function Node({ n, denom }: { n: ModelNode; denom: number }) {
  const kids = effectiveChildren(n);
  if (n.kind === "residual" && kids?.length) {
    const pct = denom ? (100 * n.params) / denom : 0;
    return (
      <div
        className="my-[5px] rounded-lg border
                   border-[color-mix(in_srgb,var(--k-residual)_35%,transparent)]
                   bg-[color-mix(in_srgb,var(--k-residual)_5%,transparent)]
                   px-2 pb-1 pt-0.5"
        title="x + body(x) — skip connection"
      >
        <div className="flex items-center gap-[7px] px-1.5 pb-0.5 pt-[3px] text-[11.5px] text-[var(--ink-2)]">
          <span className="h-2.5 w-2.5 rounded-[3px] bg-[var(--k-residual)]" />
          residual · x + body(x)
          <span className="ml-auto text-xs tabular-nums text-[var(--muted)]">
            {pct.toFixed(1)}% · {fmt(n.params)}
          </span>
        </div>
        {kids.map((c, i) => <Node key={i} n={c} denom={denom} />)}
      </div>
    );
  }
  if (!kids?.length) return <Row n={n} denom={denom} leaf />;
  return (
    <details open>
      <summary>
        <Row n={n} denom={denom} />
      </summary>
      <div className="ml-[22px] border-l border-[var(--grid)] pl-2">
        {kids.map((c, i) => <Node key={i} n={c} denom={denom} />)}
      </div>
    </details>
  );
}

function DefineCard({ define, denom }: { define: Define; denom: number }) {
  const { name, node } = define;
  return (
    <div
      className="my-3.5 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] px-3.5 py-2.5"
      id={`def-${name}`}
    >
      <div className="mb-1.5 flex items-baseline gap-2.5 border-b border-[var(--grid)] pb-2 pt-1">
        <span className="text-[15px] font-bold">{name}</span>
        <span className="text-[12.5px] text-[var(--ink-2)]">
          {node.in} → {node.out} · {node.params.toLocaleString()} params ({fmt(node.params)})
        </span>
      </div>
      <Node n={node} denom={denom} />
    </div>
  );
}

const buttonCls =
  "cursor-pointer rounded-md border border-[var(--border)] bg-[var(--surface)] " +
  "px-2.5 py-1 text-[12.5px] text-[var(--ink-2)] hover:text-[var(--ink)]";

export default function App() {
  const data = loadData();
  const denom = Math.max(...data.defines.map((d) => d.node.params));
  const treeRef = useRef<HTMLDivElement>(null);
  // Assembled at runtime so the compiler's __TITLE__ substitution in the
  // built bundle can't rewrite this check itself.
  const placeholder = ["__TITLE", "__"].join("");
  const title = document.title.includes(placeholder) ? "jixp model" : document.title;

  const setAll = (open: boolean) =>
    treeRef.current?.querySelectorAll("details").forEach((el) => (el.open = open));

  const toggleTheme = () => {
    const root = document.documentElement;
    const osDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = root.dataset.theme || (osDark ? "dark" : "light");
    root.dataset.theme = cur === "dark" ? "light" : "dark";
  };

  return (
    <div className="mx-auto max-w-[880px]">
      <header className="mb-2 flex flex-wrap items-baseline gap-4">
        <h1 className="text-lg font-semibold">{title}</h1>
        <div className="text-[28px] font-bold">
          {fmt(data.total_params)}
          <small className="ml-1.5 text-[13px] font-normal text-[var(--ink-2)]">
            params across {data.defines.length} defines
          </small>
        </div>
        <div className="ml-auto flex gap-2">
          <button className={buttonCls} onClick={() => setAll(true)}>expand all</button>
          <button className={buttonCls} onClick={() => setAll(false)}>collapse all</button>
          <button className={buttonCls} onClick={toggleTheme}>theme</button>
        </div>
      </header>
      <div ref={treeRef}>
        {data.defines.map((d) => <DefineCard key={d.name} define={d} denom={denom} />)}
      </div>
      <footer className="mt-4 text-xs text-[var(--muted)]">
        generated by jixp · one node per source block — <code>repeat</code> is a
        single node with multiplied params · meters are % of the largest define ·
        raw JSON embedded in this file
      </footer>
    </div>
  );
}
