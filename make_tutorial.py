#!/usr/bin/env python3
"""
Build TUTORIAL.md and TUTORIAL.pdf for lpf_run.py.

The narrative is written here; the option reference is introspected from
lpf_run.build_parser(), so it cannot drift from the code. Re-run after changing
the CLI:

    python make_tutorial.py            # both files
    python make_tutorial.py --md-only  # skip the PDF
"""

import argparse
import html
import os
import re
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lpf_run  # noqa: E402

CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/usr/bin/google-chrome", "/usr/bin/chromium")


# --------------------------------------------------------------------------
# option reference, introspected from the parser
# --------------------------------------------------------------------------

def _fmt_default(action):
    d = action.default
    if d is None or d is argparse.SUPPRESS:
        return ""
    if isinstance(d, bool):
        return "`true`" if d else ""
    if isinstance(d, (list, tuple)) and not d:
        return ""
    if d == "":
        return ""
    return f"`{d}`"


def _describe(action):
    """One table row: flags, value, default, help."""
    flags = ", ".join(f"`{s}`" for s in action.option_strings)
    if isinstance(action, argparse._StoreTrueAction) or \
            isinstance(action, argparse._StoreFalseAction):
        value = "flag"
    elif action.choices:
        value = " \\| ".join(f"`{c}`" for c in action.choices)
    elif action.metavar:
        value = f"`{action.metavar}`"
    elif action.nargs == 0:
        value = "flag"
    else:
        value = f"`{(action.dest or '').upper()}`"
    help_txt = " ".join((action.help or "").split())
    help_txt = help_txt.replace("|", "\\|")
    if isinstance(action, argparse._AppendAction):
        help_txt += " *(repeatable)*"
    return flags, value, _fmt_default(action), help_txt


def option_reference():
    """Markdown for every subcommand's options, grouped as the parser groups them."""
    parser = lpf_run.build_parser()
    sub = next(a for a in parser._actions
               if isinstance(a, argparse._SubParsersAction))
    order = ["prep", "coarse", "traj", "analyze", "run", "cluster",
             "rescore", "doctor", "demo"]
    seen, out = set(), []
    for name in order + [n for n in sub.choices if n not in order]:
        p = sub.choices.get(name)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        out.append(f"\n### `lpf_run.py {name}`\n")
        desc = " ".join((p.description or "").split())
        if desc:
            out.append(desc + "\n")
        for group in p._action_groups:
            rows = []
            for a in group._group_actions:
                if not a.option_strings or a.dest in ("help",):
                    continue
                rows.append(_describe(a))
            if not rows:
                continue
            title = group.title or ""
            if title in ("options", "optional arguments", "positional arguments"):
                title = "other"
            out.append(f"\n**{title}**\n")
            out.append("| option | value | default | what it does |")
            out.append("|---|---|---|---|")
            for flags, value, default, help_txt in rows:
                out.append(f"| {flags} | {value} | {default} | {help_txt} |")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# a minimal markdown -> html converter
#
# Only the constructs this document uses, so the PDF is guaranteed to match
# and nothing has to be installed.
# --------------------------------------------------------------------------

def _inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = t.replace("\\|", "|")
    return t


def md_to_html(md, title):
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            lang = ln[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre class="{html.escape(lang)}"><code>'
                       + html.escape("\n".join(body)) + "</code></pre>")
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"^\s*---+\s*$", ln):
            out.append("<hr/>")
            i += 1
            continue

        if ln.lstrip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in header)
                       + "</tr></thead><tbody>")
            for r in rows:
                r += [""] * (len(header) - len(r))
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>"
                                            for c in r[:len(header)]) + "</tr>")
            out.append("</tbody></table>")
            continue

        if ln.startswith(">"):
            body = []
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(body)) + "</blockquote>")
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = bool(re.match(r"\d", m.group(2)))
            tag = "ol" if ordered else "ul"
            items, base = [], len(m.group(1))
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if mm and len(mm.group(1)) == base:
                    txt = mm.group(3)
                    i += 1
                    while i < len(lines) and lines[i].strip() and \
                            not re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[i]) and \
                            not lines[i].startswith("```"):
                        txt += " " + lines[i].strip()
                        i += 1
                    items.append(txt)
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>"
                                            for x in items) + f"</{tag}>")
            continue

        if not ln.strip():
            i += 1
            continue

        body = []
        while i < len(lines) and lines[i].strip() \
                and not lines[i].startswith(("#", "```", "|", ">")) \
                and not re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[i]) \
                and not re.match(r"^\s*---+\s*$", lines[i]):
            body.append(lines[i].strip())
            i += 1
        if body:
            out.append("<p>" + _inline(" ".join(body)) + "</p>")

    css = """
@page { size: A4; margin: 17mm 15mm 17mm 15mm; }
body { font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1a1a1a; }
h1 { font-size: 21pt; margin: 0 0 2pt; border-bottom: 2px solid #1f77b4;
     padding-bottom: 5pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; color: #11507d;
     border-bottom: 1px solid #d8d8d8; padding-bottom: 3pt;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #11507d;
     page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; }
p { margin: 5pt 0; }
code { font: 9.3pt/1.4 "SF Mono", Menlo, Consolas, monospace;
       background: #f3f5f7; padding: 0.5pt 2.5pt; border-radius: 2px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-left: 3px solid #1f77b4;
      border-radius: 3px; padding: 7pt 9pt; margin: 7pt 0; overflow-x: auto;
      page-break-inside: avoid; }
pre code { font-size: 9pt; background: none; padding: 0; white-space: pre-wrap;
           word-break: break-word; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d8dde2; padding: 3.5pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #eef3f7; font-weight: 600; }
td code, th code { font-size: 8.4pt; }
blockquote { margin: 7pt 0; padding: 5pt 10pt; background: #fff9e6;
             border-left: 3px solid #e0b000; }
blockquote p { margin: 0; }
ul, ol { margin: 5pt 0 5pt 17pt; padding: 0; }
li { margin: 2.5pt 0; }
hr { border: none; border-top: 1px solid #e1e4e8; margin: 14pt 0; }
a { color: #11507d; }
"""
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{css}</style>"
            f"</head><body>{''.join(out)}</body></html>")


def html_to_pdf(html_path, pdf_path):
    chrome = next((c for c in CHROME if os.path.exists(c)), None)
    if chrome is None:
        print("no Chrome/Chromium found — skipping the PDF. The HTML is at "
              f"{html_path}; print it from any browser.")
        return None
    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", "--virtual-time-budget=4000",
           "file://" + html_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.isfile(pdf_path):
        print("Chrome did not produce a PDF:")
        print(textwrap.indent((proc.stderr or proc.stdout)[-1200:], "   "))
        return None
    return pdf_path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    body_path = os.path.join(HERE, "tutorial_body.md")
    if not os.path.isfile(body_path):
        sys.exit(f"missing {body_path}")
    body = open(body_path).read()
    if "{{OPTION_REFERENCE}}" not in body:
        sys.exit("tutorial_body.md has no {{OPTION_REFERENCE}} placeholder")
    md = body.replace("{{OPTION_REFERENCE}}", option_reference())

    md_path = os.path.join(HERE, "TUTORIAL.md")
    with open(md_path, "w") as fh:
        fh.write(md)
    n_opts = md.count("| `--")
    print(f"TUTORIAL.md      {len(md.splitlines()):5d} lines, "
          f"{len(md) / 1024:.0f} KiB, {n_opts} documented options")

    if "--md-only" in sys.argv:
        return 0

    html_path = os.path.join(HERE, "_tutorial.html")
    with open(html_path, "w") as fh:
        fh.write(md_to_html(md, "Ligand PathFinder from the command line"))
    pdf_path = os.path.join(HERE, "TUTORIAL.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    out = html_to_pdf(html_path, pdf_path)
    if out:
        print(f"TUTORIAL.pdf     {os.path.getsize(out) / 1024:.0f} KiB")
        os.remove(html_path)
    else:
        print(f"HTML kept at {html_path} — print it from a browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
