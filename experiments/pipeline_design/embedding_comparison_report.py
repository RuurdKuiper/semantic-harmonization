"""Concise tables and figures for the validation-only embedding extension."""
import json
import os
from pathlib import Path

import pandas as pd

from experiments.pipeline_design.report import NAMES,categorical_bars,table


def conditions(config,selections):
    fixed=config['fixed_lexical_weight']
    return [(provider,label,kind,weight) for provider,label in (('local','MiniLM'),('openai','OpenAI'))
            for kind,weight in [('50:50',fixed),('optimized',selections.get(provider,{}).get('lexical_weight'))]]


def write_report(output,config,cases):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    selections=json.loads((output/'retrieval_selection.json').read_text()) if (output/'retrieval_selection.json').exists() else {}
    retrieval=pd.read_csv(output/'retrieval.csv') if (output/'retrieval.csv').exists() else pd.DataFrame()
    pipeline=pd.read_csv(output/'pipeline_metrics.csv') if (output/'pipeline_metrics.csv').exists() else pd.DataFrame()
    planned=conditions(config,selections)
    text=['# Experiment 1 extension: MiniLM versus OpenAI embeddings','',
        'Seven validation EDFs compare the same corpus, queries, BM25 scores, and reference labels. '
        'Type 1 diabetes and published-benchmark cases are excluded. Original frozen pipelines and results are retained.','',
        'MiniLM is compared with text-embedding-3-large (3,072 dimensions) at 50:50 lexical/semantic weighting '
        'and at each embedder’s best of the same 21 lexical weights. Selection uses equally weighted EDF average precision. '
        'Jev and GPT-6 Luna are fixed to the original proprietary choices. Review uses confidence <0.3, at most 300 '
        'candidates, and medium reasoning. Adaptive stopping and the 10,000-candidate ceiling are unchanged. '
        'CodeMapper is excluded. Identical conditions share checkpoints.','',
        'Both encoders receive the same query strings; their native tokenization and input-length behavior can differ. '
        'OpenAI description vectors and query vectors are normalized, with six-decimal cosine rounding before global '
        'min–max normalization, matching the original retrieval implementation. Embeddings are checkpointed by batch.','',
        '## Retrieval selection','']
    text.append(table(['Embedder','Selected lexical weight','Mean AP'],[
        [label,f'{selections[provider]["lexical_weight"]:.2f}',f'{selections[provider]["mean_average_precision"]:.4f}']
        if provider in selections else [label,'Pending','Pending'] for provider,label in [('local','MiniLM'),('openai','OpenAI')]]))
    text+=['','## Per-EDF retrieval','']
    labels=[f'{label} {kind}' for _,label,kind,_ in planned]
    rows=[]
    for name in cases:
        cells=[]
        for provider,_,_,weight in planned:
            found=retrieval[(retrieval.embedder==provider)&(retrieval.phenotype==name)&(retrieval.lexical_weight==weight)] if not retrieval.empty and weight is not None else pd.DataFrame()
            cells.append(f'AP {found.iloc[0].average_precision:.3f}<br>R@1k {100*found.iloc[0].recall_at_1000:.1f}%<br>R@10k {100*found.iloc[0].recall_at_10000:.1f}%' if len(found) else 'Pending')
        rows.append([NAMES[name],*cells])
    text+=[table(['EDF',*labels],rows),'','## End-to-end evaluation after GPT review','']
    rows=[]
    for name in cases:
        cells=[]
        for provider,_,_,weight in planned:
            found=pipeline[(pipeline.embedder==provider)&(pipeline.phenotype==name)&(pipeline.lexical_weight==weight)&(pipeline.stage=='review')] if not pipeline.empty and weight is not None else pd.DataFrame()
            cells.append('<br>'.join(f'{tag} {100*found.iloc[0][key]:.1f}%' for tag,key in [('R','recall'),('P','precision'),('F1','f1')]) if len(found) else 'Pending')
        rows.append([NAMES[name],*cells])
    text+=[table(['EDF',*labels],rows),'',
        'All available reference Narrow codes enter recall, including unclassified positives. '
        'Averages are omitted until all seven EDFs for a condition are complete. Neither references nor evaluation results '
        'enter prompts or select model identities. Initial-decision and post-review confusion counts, candidate counts, '
        'ceiling hits, and review counts are in `pipeline_metrics.csv`. All 21 retrieval weights are in `retrieval.csv`.','']
    if not retrieval.empty:
        os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.model-cache/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
        categories=[*cases,'__average__']
        for source,filename,metrics in [(retrieval,'retrieval',[('average_precision','Average precision (%)'),('recall_at_1000','Recall at 1,000 (%)'),('recall_at_10000','Recall at 10,000 (%)')]),
                                       (pipeline,'review',[('recall','Recall (%)'),('precision','Precision (%)'),('f1','F1 (%)')])]:
            if source.empty:continue
            fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
            for ax,(metric,ylabel) in zip(axes,metrics):
                series=[]
                for k,(provider,label,kind,weight) in enumerate(planned):
                    group=source[(source.embedder==provider)&(source.lexical_weight==weight)] if weight is not None else source.iloc[:0]
                    if 'stage' in group:group=group[group.stage=='review']
                    values=group.set_index('phenotype')[metric]
                    series.append((f'{label} {kind}',values,{'color':['#2566a5','#79a7d1','#c55b13','#e5a46e'][k]}))
                categorical_bars(ax,categories,series)
                ax.set(ylabel=ylabel,ylim=(0,105));ax.grid(axis='y',alpha=.2)
            axes[0].legend(ncol=2,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.02))
            axes[-1].set(xticks=range(len(categories)),xticklabels=[NAMES.get(n,'Average') for n in categories])
            plt.setp(axes[-1].get_xticklabels(),rotation=35,ha='right');fig.tight_layout()
            fig.savefig(figures/f'{filename}.pdf');fig.savefig(figures/f'{filename}.png',dpi=180);plt.close(fig)
            text += [f'![{filename} comparison](figures/{filename}.png)','']
    usage=output/'embedding_usage.json'
    if usage.exists():
        info=json.loads(usage.read_text());text+=['## Embedding usage','',
            f'Checkpointed corpus embeddings: {info["corpus_rows"]:,} rows, {info["corpus_tokens"]:,} input tokens, '
            f'{info["corpus_requests"]:,} requests. Query usage is recorded separately.','']
    status=output/'status.json'
    if status.exists():
        info=json.loads(status.read_text());text += [f'Current phase: {info["phase"]}; status: {info["status"]}.','']
    (output/'report.md').write_text('\n'.join(text)+'\n')
