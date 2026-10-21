"""Insert a Colab setup cell at the top of executed notebooks (keeps existing outputs)."""
import sys
import nbformat

SETUP = '''# Colab set-up: run this first (skipped automatically outside Colab)
import sys, os
if "google.colab" in sys.modules:
    REPO_URL = "https://github.com/anvesha2402/prakriti-growth-measurement.git"   # change if your repo differs
    if not os.path.exists("/content/repo"):
        get_ipython().system(f"git clone {REPO_URL} /content/repo")
    os.chdir("/content/repo")
    get_ipython().system("pip install -q 'pymc-marketing==1.1.*' statsmodels h5netcdf")
'''

for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    if nb.cells and "Colab set-up" in "".join(nb.cells[1].source if len(nb.cells) > 1 else ""):
        continue
    cell = nbformat.v4.new_code_cell(SETUP)
    nb.cells.insert(1, cell)
    import uuid
    for c in nb.cells:
        c.setdefault("id", uuid.uuid4().hex[:8])
    nbformat.write(nb, path)
    print("added to", path)
