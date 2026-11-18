"""Static sanity checks for main.tex: environment balance, citation keys,
label/ref resolution, and figure file existence (compilation is not possible
in this environment because the local TinyTeX binary is broken)."""

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "latex" / "main.tex"


def main() -> None:
    src = TEX.read_text(encoding="utf-8")

    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", src))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", src))
    print("environment balance:")
    problems = 0
    for env in sorted(set(begins) | set(ends)):
        b, e = begins[env], ends[env]
        flag = "OK" if b == e else "MISMATCH"
        if b != e:
            problems += 1
        print(f"  {env:20s} begin={b} end={e} {flag}")

    bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
    # external .bib files referenced via \bibliography{...}
    for bibname in re.findall(r"\\bibliography\{([^}]+)\}", src):
        bibpath = TEX.parent / f"{bibname.strip()}.bib"
        if bibpath.exists():
            bibsrc = bibpath.read_text(encoding="utf-8")
            bibkeys |= set(re.findall(r"@\w+\{([^,]+),", bibsrc))
        else:
            print(f"  WARNING: cited bib file not found: {bibpath.name}")
    citekeys = set()
    for m in re.findall(r"\\cite[tp]?\{([^}]+)\}", src):
        citekeys.update(k.strip() for k in m.split(","))
    missing = sorted(citekeys - bibkeys)
    unused = sorted(bibkeys - citekeys)
    print(f"\ncitations: {len(citekeys)} cited, {len(bibkeys)} in bibliography")
    print(f"  missing bibitem: {missing if missing else 'none'}")
    print(f"  uncited bibitem: {unused if unused else 'none'}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", src))
    print(f"\nlabels: {len(labels)}, refs: {len(refs)}")
    print(f"  unresolved refs: {sorted(refs - labels) if refs - labels else 'none'}")

    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    print("\nfigures:")
    for f in figs:
        p = TEX.parent / f
        print(f"  {f}: {'found' if p.exists() else 'MISSING'}")

    print(f"\nplaceholders left: {re.findall(r'pending', src)}")
    print(f"document lines: {len(src.splitlines())}")

    # --- undefined-command heuristic -------------------------------------
    # commands defined in this file, plus a whitelist of common LaTeX /
    # package / class commands used by ACL-style manuscripts
    defined = set(re.findall(r"\\(?:newcommand|renewcommand|DeclareMathOperator)\*?\{?\\([A-Za-z@]+)", src))
    whitelist = {
        # structure
        "documentclass", "usepackage", "begin", "end", "title", "author", "date",
        "maketitle", "section", "subsection", "subsubsection", "paragraph",
        "appendix", "item", "caption", "label", "ref", "eqref", "centering",
        "bibliography", "bibliographystyle", "includegraphics", "multicolumn",
        "toprule", "midrule", "bottomrule", "small", "footnotesize", "scshape",
        "textbf", "emph", "texttt", "textsc", "textsuperscript", "cite",
        "citep", "citet", "toprule", "tabular", "hline", "vspace", "hspace",
        "quad", "qquad", "linewidth", "textwidth", "columnwidth", "itemsep",
        # math / symbols
        "alpha", "beta", "sigma", "in", "le", "ge", "times", "approx", "pm",
        "ell", "top", "Omega", "sum", "frac", "text", "mathrm", "mathcal",
        "mathbb", "argmax", "softmax", "eta", "theta", "mu", "Delta", "to",
        "downarrow", "uparrow", "dots", "cdot", "quad", "neg", "forall",
        "greater", "less", "logits", "yes", "no", "method", "sout",
        # spacing / typography
        "bigskip", "medskip", "noindent", "newline", "clearpage", "And",
        "copyright", "checkmark", "cmark", "xmark", "ding", "textcolor",
        "color", "url", "href", "footnote", "thanks", "and", "and",
        "newcommand", "renewcommand", "rightarrow", "Rightarrow", "textbackslash",
        "bf", "bibitem", "captionsetup", "texttt", "mathrm", "operatorname",
        "mathbf", "resizebox", "textsc", "columnwidth", "cline",
    }
    used = set(re.findall(r"\\([A-Za-z@]+)", src))
    unknown = sorted(used - defined - whitelist)
    print(f"\nundefined-command check: {len(unknown)} command(s) not defined locally or whitelisted")
    if unknown:
        print("  " + ", ".join(unknown))
    print(f"problems: {problems} environment mismatches")


if __name__ == "__main__":
    main()
