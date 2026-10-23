"""Readable Results prose derived from the same saved metrics as manuscript plots."""
import json
from pathlib import Path
import re

import pandas as pd


def percent(value,places=1):
    return f'{100*value:.{places}f}'+r'\%'


def replace_prose(text,title,prose):
    """Retain a section's existing tables and figures while replacing its prose."""
    pattern=re.compile(r'(\\sub(?:sub)?section\{'+re.escape(title)+r'\})(.*?)(?=\\sub(?:sub)?section\{|\Z)',re.S)
    def replacement(match):
        # Capture the complete environments, rather than the backreference group.
        blocks=[m[0] for m in re.finditer(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}',match[2],re.S)]
        return match[1]+'\n\n'+'\n\n'.join([*prose,*blocks])+'\n\n'
    result,count=pattern.subn(replacement,text)
    if count!=1:raise ValueError(f'Expected one Results section: {title}')
    return result


def polish_results(text,output,config):
    """Completed validation/test prose, with explicitly partial benchmark coverage."""
    output=Path(output)
    required=['decision_metrics.csv','review_metrics.csv','mapping_metrics.csv','test_metrics.csv']
    if not all((output/name).exists() for name in required):return text
    decision,review,mapping,test=[pd.read_csv(output/name) for name in required]
    n=len(config['validation_edfs'])
    if (decision.groupby('name').phenotype.nunique()!=n).any() or len(test)!=6:return text
    d=decision.groupby('name')[['recall','precision','f1']].mean()
    shared=review[review.branch=='overall'].groupby('name')[['recall','precision','f1']].mean()
    if len(shared)!=3 or review[review.branch=='open_weight'].phenotype.nunique()!=n:return text
    m=mapping.groupby(['branch','stage'])[['recall','precision','f1']].mean()
    if len(m)!=4:return text
    from experiments.pipeline_design.embedding_comparison_manuscript import completed_extension
    extension=completed_extension(output)
    # Keep the source-derived dataset table and its accounting, but combine short fragments.
    text=text.replace('Table~\\ref{tab:study-dataset} summarizes the EDF split and reference-code availability.\n\n'
                      'The terminology corpus contains 619,527 codes from 6 vocabularies.\n\n',
                      'The terminology corpus contains 619,527 codes across six vocabularies. '
                      'Table~\\ref{tab:study-dataset} summarizes the eight EDFs and reference-code availability. ')
    selected=json.loads((output/'retrieval_selection.json').read_text())
    ranking=pd.read_csv(output/'retrieval.csv').groupby('lexical_weight').average_precision.mean()
    ranking_prose=[
        'Equal lexical and MiniLM weighting gave the highest mean average precision among the 21 tested weights '
        f'({selected["mean_average_precision"]:.3f}), compared with {ranking.loc[0]:.3f} for embeddings alone and '
        f'{ranking.loc[1]:.3f} for lexical search alone. Appendix Figure A1 shows the MiniLM and OpenAI weight sweeps.']
    if extension:
        fixed=extension['retrieval'].query('lexical_weight == 0.5').groupby('embedder').average_precision.mean()
        best=extension['selection']['openai'];pipeline=extension['pipeline'].query('stage == "review"')
        def final(provider,weight):
            return pipeline[(pipeline.embedder==provider)&(pipeline.lexical_weight==weight)].f1.mean()
        ranking_prose.extend([
            'In the controlled embedding comparison, mean average precision at 50:50 lexical/semantic weighting was '
            f'{fixed["local"]:.3f} for MiniLM and {fixed["openai"]:.3f} for OpenAI embeddings '
            r'(Figure~\ref{fig:study-retrieval}). The highest OpenAI mean average precision was '
            f'{best["mean_average_precision"]:.3f}, at {100*best["lexical_weight"]:.0f}'+r'\% lexical and '+
            f'{100*(1-best["lexical_weight"]):.0f}'+r'\% semantic weighting. '
            'Appendix Figure A2 reports recall within the first 100, 1,000, and 10,000 candidates.',
            'Using the same Jev and GPT-6 Luna settings, mean post-review F1 was '
            f'{percent(final("local",.5))} with MiniLM 50:50, {percent(final("openai",.5))} with OpenAI 50:50, and '
            f'{percent(final("openai",best["lexical_weight"]))} with OpenAI at its AP-selected weight. '
            'Appendix Figure A4 and Table A14 report these per-EDF results; Appendix Table A15 gives the confusion counts. '
            'This comparison followed the original pipeline freeze. Experiments 2 and 3 therefore retain MiniLM with equal lexical and semantic weighting.'])
    else:
        ranking_prose[0]+=r' Figure~\ref{fig:study-retrieval} shows the retrieval comparison.'
    text=replace_prose(text,'Ranking',ranking_prose)
    text=replace_prose(text,'Preliminary decisions',[
        f'Mean EDF F1 was {percent(d.loc["jev","f1"],2)} for Jev, {percent(d.loc["openai_decisions","f1"],2)} '
        f'for OpenAI Decisions, and {percent(d.loc["jpt4b","f1"],2)} for JPT-4B '
        r'(Figure~\ref{fig:study-decision_models}). Jev had mean recall of '+percent(d.loc['jev','recall'])+
        ' and precision of '+percent(d.loc['jev','precision'])+'; the corresponding values were '+
        percent(d.loc['openai_decisions','recall'])+' and '+percent(d.loc['openai_decisions','precision'])+
        ' for OpenAI Decisions, and '+percent(d.loc['jpt4b','recall'])+' and '+percent(d.loc['jpt4b','precision'])+
        ' for JPT-4B. Appendix Table A8 gives the per-EDF metrics.',
        'Jev was selected for the proprietary branch and JPT-4B for the local branch. '
        'All three models reached the 10,000-candidate ceiling for kidney disease; JPT-4B also reached it for myocarditis. '
        'Laya was excluded after accepting unrelated codes in early validation runs.'])
    local=m.loc[('open_weight','review')];prop=m.loc[('proprietary','review')]
    text=replace_prose(text,'Selective review',[
        'Reviewing the same Jev decisions yielded mean F1 values of '+
        f'{percent(shared.loc["gpt6_luna","f1"])} for GPT-6 Luna, {percent(shared.loc["gpt61_sol","f1"])} '
        f'for GPT-6.1 Sol, and {percent(shared.loc["qwen35_9b_8bit","f1"])} for Qwen3.5-9B '
        r'(Figure~\ref{fig:study-review_overall}). With the selected GPT-6 Luna reviewer, mean recall changed from '+
        percent(d.loc['jev','recall'])+' to '+percent(prop.recall)+' and precision from '+
        percent(d.loc['jev','precision'])+' to '+percent(prop.precision)+'.',
        'For the local pipeline, Qwen reviewed JPT-4B decisions on all seven EDFs. Mean F1 changed from '+
        percent(d.loc['jpt4b','f1'])+' to '+percent(local.f1)+', precision from '+
        percent(d.loc['jpt4b','precision'])+' to '+percent(local.precision)+', and recall from '+
        percent(d.loc['jpt4b','recall'])+' to '+percent(local.recall)+'. '
        'Appendix Figure A3 shows this separate comparison. Appendix Tables A6 and A7 give review metrics and confusion counts.'])
    mapped_prop=m.loc[('proprietary','codemapper')];mapped_local=m.loc[('open_weight','codemapper')]
    text=replace_prose(text,'Selected branches and optional direct mapping',[
        r'Figure~\ref{fig:study-mapping_branches} shows the three evaluated stages for each branch. '
        'After direct CodeMapper expansion, proprietary mean recall changed from '+percent(prop.recall)+' to '+
        percent(mapped_prop.recall)+', precision from '+percent(prop.precision)+' to '+percent(mapped_prop.precision)+
        ', and F1 from '+percent(prop.f1)+' to '+percent(mapped_prop.f1)+'. '
        'For the local branch, recall changed from '+percent(local.recall)+' to '+percent(mapped_local.recall)+
        ', precision from '+percent(local.precision)+' to '+percent(mapped_local.precision)+
        ', and F1 from '+percent(local.f1)+' to '+percent(mapped_local.f1)+'.',
        'Mapping was excluded from both frozen branches under the validation mean-F1 selection rule. '
        'The frozen proprietary pipeline therefore uses MiniLM/BM25, Jev, and GPT-6 Luna; '
        'the local pipeline uses the same retrieval configuration, JPT-4B, and Qwen3.5-9B.'])
    t=test.set_index(['branch','stage'])
    p0=t.loc[('proprietary','decision')];p1=t.loc[('proprietary','final')]
    l0=t.loc[('open_weight','decision')];l1=t.loc[('open_weight','final')]
    text=replace_prose(text,'Experiment 2: frozen-pipeline provisional test',[
        'On type 1 diabetes, proprietary F1 changed from '+percent(p0.f1)+' after preliminary decisions to '+
        percent(p1.f1)+' after review. Recall changed from '+percent(p0.recall)+' to '+percent(p1.recall)+
        ', with precision of '+percent(p1.precision)+' after review. Local F1 changed from '+percent(l0.f1)+
        ' to '+percent(l1.f1)+', with final recall of '+percent(l1.recall)+' and precision of '+percent(l1.precision)+
        r' (Figure~\ref{fig:study-provisional_test}). Both final outputs equal their post-review outputs because mapping was excluded.',
        'Appendix Table A9 gives the confusion counts. Type 1 diabetes appeared in earlier development; '
        'these results form a retrospective provisional test.'])
    path=output/'benchmark_metrics.csv'
    if path.exists():
        benchmark=pd.read_csv(path)
        def means(frame):
            r=frame.final_required_recall;p=frame.final_acceptable_precision
            return r.mean(),p.mean(),(2*r*p/(r+p)).fillna(0).mean()
        def summary(frame):
            r,p,f=means(frame)
            return 'recall of '+percent(r)+', acceptable-code precision of '+percent(p)+', and F1 of '+percent(f)
        from experiments.matthewman_2026.experiment import PHENOTYPES
        expected=len(PHENOTYPES)*len(config['benchmark']['subsets'])*2
        prop_rows=benchmark[benchmark.branch=='proprietary'];local_rows=benchmark[benchmark.branch=='open_weight']
        prose=[f'The published-benchmark evaluation contains {len(benchmark)} of {expected} planned runs: '
               f'{len(prop_rows)} proprietary and {len(local_rows)} local runs. '
               r'Figure~\ref{fig:study-published_benchmark} compares these outputs with the published systems; '
               r'Table~\ref{tab:study-benchmark} lists the completed runs.']
        for subset,description in [('full','full corpus'),('read','Read-only corpus')]:
            a=prop_rows[prop_rows.subset==subset];b=local_rows[local_rows.subset==subset]
            sentence=f'For the {description}, the proprietary pipeline had mean '+summary(a)+f' across {len(a)} phenotypes. '
            sentence+='The local pipeline had mean '+summary(b)+f' across {len(b)} completed phenotypes.'
            if len(a)!=len(b):sentence+=' These averages cover different phenotype sets.'
            prose.append(sentence)
        baseline=output/'published_baselines/metrics.csv'
        if baseline.exists():
            published=pd.read_csv(baseline)
            gem=published[published.model=='gemini-3-pro-preview']
            prose.append('Among the rescored published systems, Gemini 3 Pro had the highest mean F1 in both corpora: '+
                         percent(means(gem[gem.subset=='full'])[2])+' for the full corpus and '+
                         percent(means(gem[gem.subset=='read'])[2])+' for the Read-only corpus. '
                         'Published means use three runs per phenotype; our pipeline results use one run.')
        prose.append('The comparison includes 168 archived submissions. Appendix Table A10 gives original whole-list grades '
                     'and harmonized rescoring, Table A11 documents reference differences, and Table A12 gives individual published run metrics. '
                     'Identifier formatting normalization affected 117 identifiers in nine submissions; Appendix Table A13 reports its effect on scoring.')
        text=replace_prose(text,'Experiment 3: published phenotype benchmark',prose)
        if len(benchmark)==expected:
            text=text.replace('Average bars are equally weighted means across completed phenotypes; panels state incomplete coverage. '
                              'They can cover different phenotype sets while evaluation is ongoing.',
                              'Average bars give each of the seven phenotypes equal weight.')
            text=text.replace(' Missing bars indicate unfinished runs of our pipelines, rather than zero performance.','')
    text=text.replace('shows the provisional diabetes comparison; unfinished results are marked pending.',
                      'shows the provisional diabetes comparison.')
    text=text.replace('Experiment 3: completed published-benchmark runs.',
                      'Experiment 3: our completed pipeline runs on the published phenotype benchmark.')
    return text


def align_main_text(source):
    """Remove statements superseded by the controlled embedding extension."""
    source=source.replace('Embedding-model alternatives are outside this comparison.',
                          'The subsequent controlled comparison evaluates OpenAI embeddings separately from this original selection.').replace(
        'recall at 200, 1,000, and 5,000 is reported separately.',
        'recall at fixed candidate budgets is reported as a secondary measure.').replace(
        'Direct CodeMapper mapping has completed 7 of seven local EDF comparisons. Mean F1 over this paired subset changed',
        'For the local pipeline, direct CodeMapper mapping changed mean validation F1').replace(
        'Direct CodeMapper mapping has completed 7 of seven proprietary EDF comparisons. Mean F1 over this paired subset changed',
        'For the proprietary pipeline, direct CodeMapper mapping changed mean validation F1').replace(
        'The completed shared-input review comparison uses the same Jev decisions for all three reviewers.',
        'The shared-input review comparison uses the same Jev decisions for all three reviewers.').replace(
        'Hybrid retrieval and constrained preliminary decisions have been evaluated on seven validation EDFs, and the shared-input reviewer comparison is complete. '
        'The validation comparisons are complete. The provisional test is complete; the published-benchmark evaluation remains unfinished. '
        'The completed workflow provides saved decisions for inspection; independent and patient-level evaluation will be needed to assess generalization and clinical validity.',
        'The evaluated workflow combines hybrid retrieval, constrained preliminary decisions, and selective review to construct auditable codelists. '
        'Direct cross-vocabulary mapping increased recall but reduced mean validation F1 under the tested settings. '
        'Independent definitions and patient-level evaluation are needed to assess generalization and clinical validity.')
    source=re.sub(r'For the local pipeline, direct CodeMapper mapping changed mean validation F1.*?A common UMLS concept',
                  'Direct mapping increased recall and reduced precision in both branches, and neither branch retained it under the mean-F1 selection rule. '
                  'A common UMLS concept',source,flags=re.S)
    return source.replace('Restricting outputs to submitted codes supplies identifier validity within the corpus.',
                          'Restricting outputs to identifiers in the loaded terminology corpus ensures identifier validity within that corpus.')
