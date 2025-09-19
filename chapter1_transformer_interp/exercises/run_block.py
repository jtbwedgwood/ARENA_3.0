# run_block.py
import argparse, re, sys, pathlib

CELL_SPLIT = re.compile(r'(?m)^\s*#\s*%%.*$')

def _line_no_from_pos(text: str, pos: int) -> int:
    # 1-based line number of byte offset pos
    return text.count("\n", 0, pos) + 1

def load_cells_with_lines(path: str):
    """
    Returns a list of dicts: {'src': <cell_source>, 'lineno': <start_line_in_original>}
    Cells are delimited by lines matching CELL_SPLIT. Any preamble before the first cell
    is also treated as a cell starting at line 1.
    """
    txt = pathlib.Path(path).read_text()
    matches = list(CELL_SPLIT.finditer(txt))
    cells = []

    # preamble before first # %% (if any)
    start = 0
    if matches:
        first = matches[0]
        if first.start() > 0:
            src = txt[start:first.start()]
            if src.strip():
                cells.append({"src": src, "lineno": 1})
        # iterate real cells
        for i, m in enumerate(matches):
            cell_start = m.end()
            cell_end = matches[i+1].start() if i+1 < len(matches) else len(txt)
            src = txt[cell_start:cell_end]
            if src.strip():
                lineno = _line_no_from_pos(txt, cell_start)  # 1-based
                cells.append({"src": src, "lineno": lineno})
    else:
        # no delimiters: whole file is one cell
        if txt.strip():
            cells.append({"src": txt, "lineno": 1})

    return cells

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="Path to a #%%-segmented python file")
    ap.add_argument("--which", default="latest",
                    help="'latest' or 0-based index of cell to run (after setup).")
    ap.add_argument("--setup-cells", type=int, default=1,
                    help="How many initial cells are 'setup' (default: 1).")
    args = ap.parse_args()

    cells = load_cells_with_lines(args.script)
    if not cells:
        print("No cells found.", file=sys.stderr)
        sys.exit(2)

    # shared exec namespace
    g = {"__name__": "__main__"}

    # run setup cells
    for i in range(min(args.setup_cells, len(cells))):
        src, lineno = cells[i]["src"], cells[i]["lineno"]
        padded = ("\n" * (lineno - 1)) + src
        code = compile(padded, args.script, "exec")
        exec(code, g)

    # pick target cell
    if args.which == "latest":
        idx = len(cells) - 1
    else:
        idx = int(args.which)
        idx = args.setup_cells + idx
        idx = min(max(idx, args.setup_cells), len(cells) - 1)

    src, lineno = cells[idx]["src"], cells[idx]["lineno"]
    padded = ("\n" * (lineno - 1)) + src
    target = compile(padded, args.script, "exec")
    exec(target, g)

if __name__ == "__main__":
    main()
