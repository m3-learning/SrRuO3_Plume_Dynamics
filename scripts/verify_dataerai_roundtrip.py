#!/usr/bin/env python3
"""Opt-in live check: downloaded input -> cell outputs -> saved SVG -> provenance.

Uses historical, already reported thickness values, clearly labelled as such.
Keeps validation execution outputs out of the original scientific notebooks.
"""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    destination = ROOT / '.dataerai/validation/roundtrip.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / 'notebooks/1_Abstract.ipynb').read_text())
    setup = next(c for c in source['cells'] if 'dataerai-managed-setup' in c['metadata'].get('tags', []))
    setup_source = ''.join(setup['source']).replace('notebooks/1_Abstract.ipynb', '.dataerai/validation/roundtrip.ipynb')
    check_source = '''%%dataerai
%matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

workbook = pd.read_excel('../data/Growth_Parameters.xlsx')
display(workbook)
reported = json.loads((REPO_ROOT / 'preservation/key_results.json').read_text())
validation_results = pd.DataFrame(reported['film_thickness']['rows'])
display(validation_results)
validation_values = validation_results['thickness_mean_nm'].to_numpy()
fig, ax = plt.subplots(figsize=(6, 3))
ax.errorbar(validation_results['sample_id'], validation_values,
            yerr=validation_results['thickness_std_nm'], fmt='o')
ax.set(xlabel='Sample ID', ylabel='Reported thickness (nm)',
       title='Preservation check: historical saved results')
fig.tight_layout()
fig.savefig('../figures/dataerai-validation-thickness.svg')
plt.show()
print('Verified downloaded workbook and preserved a table, array, plot and SVG.')
'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Dataerai preservation validation\n\nHistorical results; no scientific re-analysis.'),
        nbformat.v4.new_code_cell(setup_source),
        nbformat.v4.new_code_cell(check_source),
        nbformat.v4.new_code_cell('dataerai.finish()'),
    ])
    nbformat.write(notebook, destination)
    client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
    error_outputs = [o for c in notebook.cells for o in c.get('outputs', []) if o.output_type == 'error']
    assert not error_outputs
    print(f'Live validation completed: {destination}')
    for c in notebook.cells:
        for output in c.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text)
    return destination


if __name__ == '__main__':
    main()
