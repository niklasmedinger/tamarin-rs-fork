#!/usr/bin/env python3
"""Helpers of the canonicalization pin gate (scripts/canon_pin.sh):
normalize and compare `explore_canonical_matches` JSON (schema 4), and pick
a tier's lemma list from a calibration run.

  canon_pin.py normalize <in.json> <out.json[.gz]>
      Drop everything that depends on the machine, the run or the path
      (timings, memory, argv, the theory path, the nondeterministic budget
      parameters) and write the rest -- the stored graph.

  canon_pin.py compare <base.json[.gz]> <branch.json[.gz]>
      Print one line `STATUS<TAB>detail` (both sides are normalized first):

        SAME               identical
        FP_ONLY            identical but for per-node canonical fingerprints:
                           the canonical FORM changed, the merges did not
                           (info, not a failure)
        MERGE_DIFF         a node offers the same methods with the same cases,
                           but a case leads to a different node: a system was
                           merged on one side only, or into a different node
                           (canonicalization)
        CANON_FAIL_DIFF    canonicalization failed (error/panic) on one side only
        SOLVER_DIFF        a node's status, result or methods differ (the
                           solver or the ranking, not canonicalization)
        STOP_DIFF          a node expanded on one side is still queued on the
                           other, or only stop_reason/sizes differ
        METHOD_CHECK_DIFF  only the merge check's counts differ
        OTHER_DIFF         anything else (lists the differing keys)

      The walk goes over the nodes in id order (BFS creation order, so two
      identical runs number their nodes identically) and reports the FIRST
      node that differs, with its depth and its path from the root
      (`method [case] -> ...`), which can be followed in the GUI.

  canon_pin.py select <fast|full> <calibration.tsv> <corpus-dir> <budget-secs>
      Print a tier's lemma list (`relpath lemma  # secs graph family` rows)
      from a calibration run's rows (relpath, lemma, status, secs, graph,
      stop_reason). Only lemmas that ran to a JSON with at least
      MIN_GRAPH nodes are candidates. `budget-secs` is the summed
      single-job run time the list may cost.
        fast  round-robin over feature families (SAPIC, accountability,
              -D, auto-sources, DH, xor, multiset, bilinear, natural
              numbers, loops, plain), cheapest lemma of a not-yet-picked
              theory first, until the budget is spent.
        full  every candidate; if over budget, the most expensive are
              dropped (and listed in the header).

  canon_pin.py --selftest
"""

import copy
import gzip
import json
import os
import re
import sys

# Top-level keys that depend on the machine, the run or where the theory
# lives, never on what the exploration found.
VOLATILE_TOP = ("timing", "peak_rss_mib", "argv", "theory")
# Budgets that stop a run at a machine-dependent point; the gate never
# passes them, and their values say nothing about the graph.
VOLATILE_PARAMS = ("time_budget", "max_rss_gb", "setup_timeout")
VOLATILE_NODE = ("canon_secs", "methods_secs", "exec_secs")

FAILING = ("MERGE_DIFF", "CANON_FAIL_DIFF", "SOLVER_DIFF", "STOP_DIFF",
           "METHOD_CHECK_DIFF", "OTHER_DIFF")


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def normalize(doc):
    doc = copy.deepcopy(doc)
    for k in VOLATILE_TOP:
        doc.pop(k, None)
    for k in VOLATILE_PARAMS:
        doc.get("params", {}).pop(k, None)
    for n in doc.get("nodes", []):
        for k in VOLATILE_NODE:
            n.pop(k, None)
    return doc


def without_fingerprints(doc):
    doc = copy.deepcopy(doc)
    for n in doc.get("nodes", []):
        n.pop("fingerprint", None)
    return doc


def short(s, limit=100):
    s = " ".join(str(s).split())
    return s if len(s) <= limit else s[:limit] + "…"


def path_of(doc, node_id):
    """`method [case] -> ...` from the root to `node_id`, via first_parent."""
    nodes, methods = doc["nodes"], doc["methods"]
    steps = []
    cur = nodes[node_id].get("first_parent")
    seen = 0
    while cur is not None and seen <= len(nodes):
        parent, and_idx, case_idx = cur
        a = nodes[parent]["and"][and_idx]
        steps.append(f"{short(methods[a['method']], 60)} [{a['cases'][case_idx][0]}]")
        cur = nodes[parent].get("first_parent")
        seen += 1
    return " -> ".join(reversed(steps)) or "<root>"


def method_rows(doc, node):
    """The node's applied methods as (text, kind, case names), in order."""
    return [(doc["methods"][a["method"]], a["kind"], tuple(c[0] for c in a["cases"]))
            for a in node.get("and", [])]


def multiset_diff(a, b):
    rest = list(b)
    only_a = []
    for x in a:
        if x in rest:
            rest.remove(x)
        else:
            only_a.append(x)
    return only_a, rest


def is_new_child(doc, parent, and_idx, case_idx, child):
    return doc["nodes"][child].get("first_parent") == [parent, and_idx, case_idx]


def compare_node(base, branch, i):
    """None if node i is the same on both sides, else (status, detail)."""
    nb, nr = base["nodes"][i], branch["nodes"][i]
    where = f"node {i} depth {nb.get('depth')} path {path_of(base, i)}"

    sb, sr = nb.get("status"), nr.get("status")
    canon_b = nb.get("canon_err") or sb == "canon_panic"
    canon_r = nr.get("canon_err") or sr == "canon_panic"
    if canon_b != canon_r:
        return ("CANON_FAIL_DIFF",
                f"canonicalization failed only in {'base' if canon_b else 'branch'} "
                f"(status {sb}/{sr}) at {where}")
    if sb != sr or nb.get("result") != nr.get("result"):
        kind = "STOP_DIFF" if "unexpanded" in (sb, sr) else "SOLVER_DIFF"
        return (kind, f"status base={sb}{'/' + str(nb.get('result')) if nb.get('result') else ''} "
                      f"branch={sr}{'/' + str(nr.get('result')) if nr.get('result') else ''} at {where}")

    rows_b, rows_r = method_rows(base, nb), method_rows(branch, nr)
    if rows_b != rows_r:
        only_b, only_r = multiset_diff(rows_b, rows_r)
        if not only_b and not only_r:
            return ("SOLVER_DIFF", f"same methods in a different order at {where}")
        fmt = lambda rs: "; ".join(f"{short(t, 80)} {list(c)}" for t, _, c in rs[:3]) + (
            f" (+{len(rs) - 3})" if len(rs) > 3 else "")
        return ("SOLVER_DIFF", f"methods only in base: [{fmt(only_b)}] only in branch: "
                               f"[{fmt(only_r)}] at {where}")
    for k in ("inapplicable", "untried"):
        if nb.get(k) != nr.get(k):
            return ("SOLVER_DIFF", f"{k} base={nb.get(k)} branch={nr.get(k)} at {where}")

    for ai, (ab, ar) in enumerate(zip(nb.get("and", []), nr.get("and", []))):
        for ci, ((case, cb), (_, cr)) in enumerate(zip(ab["cases"], ar["cases"])):
            if cb == cr:
                continue
            new_b = is_new_child(base, i, ai, ci, cb)
            new_r = is_new_child(branch, i, ai, ci, cr)
            if new_b and not new_r:
                how = f"merged only in branch (into node {cr})"
            elif new_r and not new_b:
                how = f"merged only in base (into node {cb})"
            elif not new_b and not new_r:
                how = f"merged into different nodes (base {cb}, branch {cr})"
            else:
                how = f"new node on both sides but numbered {cb}/{cr}"
            return ("MERGE_DIFF", f"{how}: case [{case}] of "
                                  f"{short(base['methods'][ab['method']], 80)} at {where}")

    if without_fingerprints({"nodes": [nb]}) != without_fingerprints({"nodes": [nr]}):
        keys = sorted(k for k in set(nb) | set(nr) if k != "fingerprint" and nb.get(k) != nr.get(k))
        return ("OTHER_DIFF", f"node fields {keys} at {where}")
    return None


def summary_bits(base, branch):
    bits = []
    for label, get in (
        ("graph", lambda d: d.get("sizes", {}).get("graph")),
        ("processed", lambda d: d.get("sizes", {}).get("processed")),
        ("canon_failures", lambda d: d.get("canonicalize", {}).get("failures")),
        ("method_mismatches", lambda d: d.get("method_check", {}).get("mismatches")),
        ("stop", lambda d: d.get("stop_reason")),
    ):
        vb, vr = get(base), get(branch)
        if vb != vr:
            bits.append(f"{label} {vb}->{vr}")
    return (" [" + ", ".join(bits) + "]") if bits else ""


def compare(base, branch):
    base, branch = normalize(base), normalize(branch)
    if base == branch:
        return ("SAME", f"graph={base.get('sizes', {}).get('graph')}")
    sb, sr = without_fingerprints(base), without_fingerprints(branch)
    if sb == sr:
        n = sum(1 for x, y in zip(base["nodes"], branch["nodes"])
                if x.get("fingerprint") != y.get("fingerprint"))
        return ("FP_ONLY", f"{n} of {len(base['nodes'])} node fingerprint(s) differ; merges identical")

    bits = summary_bits(base, branch)
    for i in range(min(len(base["nodes"]), len(branch["nodes"]))):
        found = compare_node(base, branch, i)
        if found:
            return (found[0], found[1] + bits)
    if len(base["nodes"]) != len(branch["nodes"]):
        return ("STOP_DIFF", f"node count {len(base['nodes'])}->{len(branch['nodes'])}{bits}")

    # Every node agrees: only top-level fields differ.
    keys = sorted(k for k in set(sb) | set(sr) if sb.get(k) != sr.get(k))
    if "canonicalize" in keys:
        return ("CANON_FAIL_DIFF", f"canonicalize {sb.get('canonicalize', {}).get('failures')}"
                                   f"->{sr.get('canonicalize', {}).get('failures')}{bits}")
    if "method_check" in keys:
        mb, mr = sb.get("method_check", {}), sr.get("method_check", {})
        return ("METHOD_CHECK_DIFF", "method_check " + ", ".join(
            f"{k} {mb.get(k)}->{mr.get(k)}" for k in ("checked", "mismatches", "failures")
            if mb.get(k) != mr.get(k)) + bits)
    if set(keys) <= {"stop_reason", "truncated", "lower_bound", "sizes", "status_counts"}:
        return ("STOP_DIFF", f"top-level {keys}{bits}")
    return ("OTHER_DIFF", f"top-level {keys}{bits}")


def dump(doc, out):
    opener = gzip.open if out.endswith(".gz") else open
    with opener(out, "wt", encoding="utf-8") as f:
        json.dump(doc, f, sort_keys=True, separators=(",", ":"))


# --- tier selection (calibrate) ------------------------------------------------

# A lemma whose exploration stays this small (e.g. solved by `simplify` at
# the root) exercises no merging, so it pins nothing.
MIN_GRAPH = 3

BUILTIN_FAMILIES = (
    ("diffie-hellman", "dh"), ("xor", "xor"), ("multiset", "multiset"),
    ("bilinear-pairing", "bilinear"), ("natural-numbers", "natural-numbers"),
)


def families(corpus, rel, flags):
    """The feature families a theory belongs to, most specific first."""
    try:
        with open(os.path.join(corpus, rel), encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    fams = []
    if re.search(r"^\s*process\s*:", text, re.M):
        fams.append("sapic")
    if "accounts for" in text:
        fams.append("accountability")
    if "-D" in flags.split() or any(x.startswith("-D=") for x in flags.split()):
        fams.append("defines")
    if "--auto-sources" in flags:
        fams.append("auto-sources")
    builtins = " ".join(re.findall(r"^\s*builtins\s*:(.*)$", text, re.M))
    fams += [name for key, name in BUILTIN_FAMILIES if key in builtins]
    if "loops/" in rel or "[use_induction" in text:
        fams.append("induction")
    return fams or ["plain"]


def read_calibration(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            parts += [""] * (7 - len(parts))
            rel, lemma, status, secs, graph, stop, flags = parts[:7]
            rows.append({"rel": rel, "lemma": lemma, "status": status,
                         "secs": float(secs or 0), "graph": int(graph or 0) if graph.isdigit() else 0,
                         "stop": stop, "flags": flags})
    return rows


def select(tier, rows, corpus, budget):
    """(picked rows, header notes) for `tier`."""
    usable = [r for r in rows if r["status"] == "OK" and r["graph"] >= MIN_GRAPH]
    notes = [f"{sum(1 for r in rows if r['status'] != 'OK')} lemma(s) left out: no JSON "
             f"(timeout or error)",
             f"{sum(1 for r in rows if r['status'] == 'OK' and r['graph'] < MIN_GRAPH)} lemma(s) "
             f"left out: graph < {MIN_GRAPH} nodes"]
    for r in usable:
        r["families"] = families(corpus, r["rel"], r["flags"])
    if tier == "full":
        picked = sorted(usable, key=lambda r: r["secs"])
        dropped = []
        while picked and sum(r["secs"] for r in picked) > budget:
            dropped.append(picked.pop())
        if dropped:
            notes.append(f"{len(dropped)} lemma(s) dropped as over budget:")
            notes += [f"  {r['rel']} {r['lemma']} ({r['secs']:.0f}s)" for r in reversed(dropped)]
        return sorted(picked, key=lambda r: (r["rel"], r["lemma"])), notes

    by_family = {}
    for r in sorted(usable, key=lambda r: (r["secs"], r["rel"], r["lemma"])):
        by_family.setdefault(r["families"][0], []).append(r)
    # Level L: round-robin over the families, each taking its cheapest
    # candidate from a theory that has exactly L lemmas picked, until no
    # family can add one; then L + 1. So every theory gets a first lemma
    # before any gets a second.
    picked, spent, per_theory = [], 0.0, {}
    level = 0
    while any(not r.get("picked") and per_theory.get(r["rel"], 0) >= level for r in usable):
        added = True
        while added:
            added = False
            for fam in sorted(by_family):
                for r in by_family[fam]:
                    if r.get("picked") or per_theory.get(r["rel"], 0) != level:
                        continue
                    if spent + r["secs"] > budget:
                        break
                    r["picked"] = True
                    picked.append(r)
                    spent += r["secs"]
                    per_theory[r["rel"]] = per_theory.get(r["rel"], 0) + 1
                    added = True
                    break
        level += 1
    counts = {}
    for r in picked:
        counts[r["families"][0]] = counts.get(r["families"][0], 0) + 1
    notes.append("by family: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return sorted(picked, key=lambda r: (r["rel"], r["lemma"])), notes


def select_main(tier, calib, corpus, budget):
    rows = read_calibration(calib)
    picked, notes = select(tier, rows, corpus, float(budget))
    total = sum(r["secs"] for r in picked)
    print(f"# {len(picked)} lemma(s) of {len({r['rel'] for r in picked})} theories, "
          f"{total:.0f}s summed single-job run time (budget {float(budget):.0f}s)")
    for n in notes:
        print(f"# {n}")
    for r in picked:
        print(f"{r['rel']} {r['lemma']}  # {r['secs']:.1f}s graph={r['graph']} "
              f"{','.join(r.get('families') or families(corpus, r['rel'], r['flags']))}")


# --- self test ---------------------------------------------------------------

def _doc():
    """root -m0-> [a: n1, b: n2]; n1 -m1-> [c: n3]; n2 -m1-> [c: n3 (merged)]."""
    node = lambda i, d, fp, parent, status="expanded", ands=(): {
        "id": i, "depth": d, "status": status, "canon_err": False, "fingerprint": fp,
        "first_parent": parent, "inapplicable": 0, "untried": 0, "canon_secs": 0.1,
        "methods_secs": 0.1, "exec_secs": 0.1, "and": list(ands)}
    return {
        "schema_version": 4, "theory": "/x/t.spthy", "lemma": "l", "argv": ["x"],
        "timing": {"setup_secs": 1}, "peak_rss_mib": 5,
        "params": {"max_depth": 4, "max_nodes": 100, "time_budget": None},
        "stop_reason": "exhausted", "sizes": {"graph": 4, "processed": 5},
        "canonicalize": {"failures": 0, "examples": []},
        "method_check": {"checked": 1, "mismatches": 0, "failures": 0, "examples": []},
        "methods": ["m0", "m1"],
        "nodes": [
            node(0, 0, "f0", None, ands=[{"method": 0, "kind": "solve", "cases": [["a", 1], ["b", 2]]}]),
            node(1, 1, "f1", [0, 0, 0], ands=[{"method": 1, "kind": "solve", "cases": [["c", 3]]}]),
            node(2, 1, "f2", [0, 0, 1], ands=[{"method": 1, "kind": "solve", "cases": [["c", 3]]}]),
            node(3, 2, "f3", [1, 0, 0], status="finished"),
        ],
    }


def selftest():
    base = _doc()
    checks = []

    other = _doc()
    other["timing"] = {"setup_secs": 99}
    other["nodes"][2]["exec_secs"] = 7
    checks.append(("volatile fields only", compare(base, other)[0], "SAME"))

    other = _doc()
    other["nodes"][3]["fingerprint"] = "zz"
    checks.append(("fingerprint only", compare(base, other)[0], "FP_ONLY"))

    # Branch fails to merge n2's child into n3: a new node 4.
    other = _doc()
    other["nodes"][2]["and"][0]["cases"] = [["c", 4]]
    other["nodes"].append(dict(other["nodes"][3], id=4, first_parent=[2, 0, 0]))
    status, detail = compare(base, other)
    checks.append(("unmerged child", status, "MERGE_DIFF"))
    checks.append(("unmerged child detail", "merged only in base" in detail, True))

    other = _doc()
    other["methods"][1] = "m1-changed"
    checks.append(("method text", compare(base, other)[0], "SOLVER_DIFF"))

    other = _doc()
    other["nodes"][3]["canon_err"] = True
    checks.append(("canon failure", compare(base, other)[0], "CANON_FAIL_DIFF"))

    other = _doc()
    other["method_check"]["mismatches"] = 1
    checks.append(("method check", compare(base, other)[0], "METHOD_CHECK_DIFF"))

    other = _doc()
    other["nodes"][3]["status"] = "unexpanded"
    checks.append(("queued vs finished", compare(base, other)[0], "STOP_DIFF"))

    checks.append(("path", path_of(base, 3), "m0 [a] -> m1 [c]"))

    # Selection: the budget (5s) fits a.l1 plus either a.l2 or b.l1; a new
    # theory goes first, and never a timeout or a trivial graph.
    row = lambda rel, lemma, secs, status="OK", graph=10: {
        "rel": rel, "lemma": lemma, "status": status, "secs": secs, "graph": graph,
        "stop": "exhausted", "flags": ""}
    rows = [row("a.spthy", "l1", 1), row("a.spthy", "l2", 2), row("b.spthy", "l1", 3),
            row("c.spthy", "l1", 1, status="TIMEOUT"), row("d.spthy", "l1", 1, graph=1),
            row("e.spthy", "l1", 50)]
    fake_family = {"a.spthy": ["plain"], "b.spthy": ["plain"], "e.spthy": ["sapic"]}
    global families
    real_families = families
    families = lambda corpus, rel, flags: fake_family.get(rel, ["plain"])
    try:
        picked, _ = select("fast", [dict(r) for r in rows], "/nonexistent", 5)
        checks.append(("fast selection", [(r["rel"], r["lemma"]) for r in picked],
                       [("a.spthy", "l1"), ("b.spthy", "l1")]))
        picked, notes = select("full", [dict(r) for r in rows], "/nonexistent", 10)
        checks.append(("full selection", [(r["rel"], r["lemma"]) for r in picked],
                       [("a.spthy", "l1"), ("a.spthy", "l2"), ("b.spthy", "l1")]))
    finally:
        families = real_families

    bad = [(name, got, want) for name, got, want in checks if got != want]
    for name, got, want in bad:
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    print(f"selftest: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


def main(argv):
    if argv[1:] == ["--selftest"]:
        return selftest()
    if len(argv) == 4 and argv[1] == "normalize":
        dump(normalize(load(argv[2])), argv[3])
        return 0
    if len(argv) == 4 and argv[1] == "compare":
        status, detail = compare(load(argv[2]), load(argv[3]))
        print(f"{status}\t{' '.join(detail.split())}")
        return 0
    if len(argv) == 6 and argv[1] == "select" and argv[2] in ("fast", "full"):
        select_main(argv[2], argv[3], argv[4], argv[5])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
