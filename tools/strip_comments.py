# Helper script to remove comments from Python files using tokenize
import io
import tokenize
from pathlib import Path

TARGETS = ["main.py", "inspect_bytes.py", "parse_main_syntax.py"]

for fname in TARGETS:
    p = Path(fname)
    if not p.exists():
        print(f"skip {fname} (not found)")
        continue
    src = p.read_bytes()
    try:
        tokens = tokenize.tokenize(io.BytesIO(src).readline)
    except Exception as e:
        print("tokenize error", fname, e)
        continue
    out_parts = []
    prev_end = (1, 0)
    for tok in tokens:
        ttype = tok.type
        tstring = tok.string
        start = tok.start
        end = tok.end
        if ttype == tokenize.ENCODING or ttype == tokenize.NL:
            continue
        if ttype == tokenize.COMMENT:
            continue
        srow, scol = start
        prow, pcol = prev_end
        if srow > prow:
            out_parts.append('\n' * (srow - prow))
            out_parts.append(' ' * scol)
        else:
            out_parts.append(' ' * (scol - pcol))
        out_parts.append(tstring)
        prev_end = end
    cleaned = ''.join(out_parts)
    if not cleaned.endswith('\n'):
        cleaned += '\n'
    out_file = p.with_suffix(p.suffix + '.cleaned')
    out_file.write_text(cleaned, encoding='utf-8')
    print('wrote', out_file)
