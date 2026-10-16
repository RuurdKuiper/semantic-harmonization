"""Recount the local retrieval corpus and render a slide-ready comparison."""
from pathlib import Path
import csv
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/semantic-harmonization-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.code_systems import CODE_SYSTEM_LOADERS, DEFAULT_CODE_SYSTEM_PATHS
from src.data.preprocessing import preprocess_corpus

OUT = Path(__file__).resolve().parent
rows = []
for vocab, loader in CODE_SYSTEM_LOADERS.items():
    source = ROOT / DEFAULT_CODE_SYSTEM_PATHS[vocab]
    loaded = loader(source)
    clean = preprocess_corpus(loaded)
    rows.append(dict(vocabulary=vocab, source=source.name,
                     loaded_unique_codes=len(loaded), retrieval_codes=len(clean)))
rows.sort(key=lambda row: row['retrieval_codes'], reverse=True)
with (OUT / 'vocabulary_counts.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

labels = {
    'SNOMEDCT_US': ('SNOMED CT US', '2025-09-01', 'Clinical concepts', 'Rich relationships; multiple parents'),
    'ICD10CM': ('ICD-10-CM', '2026', 'US diagnosis classification', 'More detail than ICD-9-CM'),
    'RCD2': ('Read v2', '2020-04-01*', 'Legacy UK clinical records', 'Last terminology update: April 2016'),
    'MDR': ('MedDRA', '28.0', 'Medical-product safety', 'Five levels; standard safety queries'),
    'ICD9CM': ('ICD-9-CM', '2014', 'Legacy US classification', 'Diagnosis + procedure codes in this file'),
    'ICPC': ('ICPC', '1993', 'Primary care classification', 'Reasons for encounter, symptoms, diagnoses'),
}
colors = {'SNOMEDCT_US':'#167D9A', 'ICD10CM':'#5268AC', 'RCD2':'#83919D',
          'MDR':'#AF6B33', 'ICD9CM':'#5268AC', 'ICPC':'#558566'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42})
fig = plt.figure(figsize=(13.33, 7.5), facecolor='white')
ink, muted = '#192C3D', '#607080'
fig.text(.045, .925, 'Six vocabularies, different purposes and scales', fontsize=23,
         fontweight='bold', color=ink)
total = sum(r['retrieval_codes'] for r in rows)
fig.text(.045, .875, f'{total:,} distinct code–vocabulary pairs in the project’s retrieval corpus',
         fontsize=13, color=muted)
ax = fig.add_axes([.225,.23,.335,.57])
ax.set_xlim(0, 410000)
ax.set_ylim(-.65, 5.65)
ax.invert_yaxis()
ax.set_xticks([0,100000,200000,300000,400000], ['0','100k','200k','300k','400k'])
ax.tick_params(axis='x', labelsize=10, colors=muted, length=0, pad=8)
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis='x', color='#E5EAEE', linewidth=.8)
ax.set_axisbelow(True)
ax.set_xlabel('Number of codes · linear scale', color=muted, fontsize=10, labelpad=12)
fig.text(.625,.815,'PRIMARY PURPOSE / DISTINCTIVE FEATURE',fontsize=9,color=muted,fontweight='bold')
for i, row in enumerate(rows):
    key = row['vocabulary']
    label, release, purpose, detail = labels[key]
    n = row['retrieval_codes']
    ax.barh(i,n,height=.38,color=colors[key],zorder=3)
    y = fig.transFigure.inverted().transform(ax.transData.transform((0,i)))[1]
    fig.text(.045,y+.009,label,fontsize=15,fontweight='bold',color=ink,va='center')
    fig.text(.045,y-.025,f'File version: {release}',fontsize=9,color=muted,va='center')
    ax.text(n+5500,i,f'{n:,}',va='center',fontsize=12,fontweight='bold',color=ink)
    fig.text(.625,y+.009,purpose,fontsize=13,fontweight='bold',color=ink,va='center')
    fig.text(.625,y-.025,detail,fontsize=10.5,color=muted,va='center')
fig.text(.045,.115,'Size reflects the local exports, not each terminology’s complete published size.',
         fontsize=11,color=ink,fontweight='bold')
fig.text(.045,.079,'Counts use the project loader and preprocessing: nonempty descriptions, deduplicated codes, grouping ranges removed.',
         fontsize=9,color=muted)
fig.text(.045,.050,'*Read v2 file identifier; the terminology was last updated in 2016. Sources and version caveats: vocabulary_notes.md.',
         fontsize=9,color=muted)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT / f'vocabulary_comparison.{ext}',dpi=240,facecolor='white')
plt.close(fig)
print(f'Created chart and counts in {OUT}; total = {total:,}')
