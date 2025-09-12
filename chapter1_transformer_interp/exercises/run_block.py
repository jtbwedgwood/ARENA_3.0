# run_block.py
import argparse, re, runpy, sys, types, os, pathlib

CELL_SPLIT = re.compile(r'(?m)^\s*#\s*%%.*$')

def load_cells(path: str):
    txt = pathlib.Path(path).read_text()
    # drop any leading empties from split
    parts = [p for p in CELL_SPLIT.split(txt) if p.strip() != ""]
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="Path to a #%%-segmented python file")
    ap.add_argument("--which", default="latest",
                    help="'latest' or 0-based index of cell to run (after setup).")
    ap.add_argument("--setup-cells", type=int, default=1,
                    help="How many initial cells are 'setup' (default: 1).")
    args = ap.parse_args()

    cells = load_cells(args.script)
    if not cells:
        print("No cells found.", file=sys.stderr)
        sys.exit(2)

    # execution namespace shared across cells
    g = {"__name__": "__main__"}  # make MAIN = True if defined in setup
    # run setup cells
    for i in range(min(args.setup_cells, len(cells))):
        code = compile(cells[i], args.script, "exec")
        exec(code, g)

    # pick target cell
    if args.which == "latest":
        idx = len(cells) - 1
    else:
        idx = int(args.which)
        # interpret index relative to post-setup cells
        idx = args.setup_cells + idx
        idx = min(max(idx, args.setup_cells), len(cells) - 1)

    target = compile(cells[idx], args.script, "exec")
    exec(target, g)

if __name__ == "__main__":
    main()
