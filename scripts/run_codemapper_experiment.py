"""Run all EDFs once and compare paired Jev, GPT, and CodeMapper codelists."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import run_pipeline
from src.data.loaders import load_aesi_dataset
from src.data.preprocessing import standardize_code
from src.evaluation.metrics import to_predicted_codelist
from src.utils.config import load_config


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confusion_tables(summaries):
    """End-to-end positive counts and assessed-negative counts, by stage."""
    lines = ["TN counts classified reference-negative codes left non-Narrow; "
             "unexamined reference-negative codes are excluded. FN includes all missed "
             "available gold Narrow codes, including unclassified codes. Gold Possible "
             "and codes absent from the reference are negative; unscored vocabularies "
             "and unavailable gold codes are excluded. The evaluated set is the union "
             "of each stage's scored decisions and available gold Narrow codes, so its "
             "size can increase when CodeMapper adds decisions.", ""]
    for stage, label in (('jev', 'Jev classification'), ('gpt', 'GPT review'),
                         ('codemapper', 'CodeMapper expansion')):
        lines += [f"**{label}**", "", "| EDF | TP | TN | FP | FN |",
                  "| --- | --- | --- | --- | --- |"]
        rows = [row for row in summaries if row['stage'] == stage]
        for row in rows:
            lines.append(f"| {row['phenotype']} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} |")
        totals = [sum(row[key] for row in rows) for key in ('tp', 'tn', 'fp', 'fn')]
        lines.append("| **Total** | " + " | ".join(str(n) for n in totals) + " |")
        lines.append("")
    return lines


def write_report(directory, manifest, summaries, details):
    retrieval = manifest['config']['retrieval']
    uncertainty = manifest['config']['uncertainty']
    llm = manifest['config']['llm']
    mapping_config = manifest['config']['codemapper']
    gold_available = sum(d['metrics']['stages']['jev']['narrow_loss_breakdown']['gold_narrow_available']
                         for d in details.values())
    gold_unavailable = sum(d['metrics']['evaluation_scope'].get('gold_narrow_excluded_as_unavailable',0)
                           for d in details.values())
    lines = ["# EDF CodeMapper experiment", "",
             "Paired, single-run comparison: Jev → GPT review → direct CodeMapper expansion. "
             "No descendants, recursive expansion, or additional LLM classification of mapped codes.", "",
             f"Retrieval: {retrieval['embedding_model']}; "
             f"{retrieval['lexical_weight']:.0%} BM25 / {retrieval['embedding_weight']:.0%} embeddings; "
             "complete corpus ranked; adaptive stopping after "
             f"{llm['consecutive_sparse_batches']} consecutive 20-code batches with at most "
             f"{llm['sparse_narrow_threshold']} Narrow decisions.", "",
             f"GPT: {llm['openai_model']}, "
             f"{'medium' if uncertainty['gpt_review_reasoning_enabled'] else 'none'} reasoning, "
             f"confidence strictly below {uncertainty['confidence_threshold']}, "
             f"at most {uncertainty['gpt_review_max_candidates']} candidates per EDF, lowest confidence first. "
             "CodeMapper seeds are frozen post-review Narrow decisions; same-vocabulary matches "
             "are ignored. Cross-vocabulary matches in the loaded corpus are added as Narrow, "
             "including promotion of existing Exclude decisions. Their confidence is unassessed "
             "(stored as 0) and they are flagged for review. Unavailable codes are recorded but not added.", "",
             "All six vocabularies participate in retrieval/mapping. Evaluation is restricted "
             "per EDF to vocabularies represented in its expert codelist and gold codes present "
             "in the loaded corpus. Gold Possible is negative in this Narrow-only experiment. "
             "Metrics are end-to-end Narrow codelist recall, precision, and F1; "
             "unclassified gold Narrow codes count as misses. "
             f"The scored denominator contains {gold_available} gold Narrow codes; "
             f"{gold_unavailable} additional gold Narrow codes were unavailable and excluded.", "",
             "| EDF | Jev R / P / F1 | GPT R / P / F1 | CodeMapper R / P / F1 | Mapping ΔTP / ΔFP |",
             "| --- | --- | --- | --- | --- |"]
    def triplet(row):
        return " / ".join(f"{100*row[k]:.1f}%" for k in ("recall", "precision", "f1"))
    for phenotype in details:
        rows = {r['stage']: r for r in summaries if r['phenotype'] == phenotype}
        before, after = rows['gpt'], rows['codemapper']
        lines.append(f"| {phenotype} | {triplet(rows['jev'])} | {triplet(before)} | "
                     f"{triplet(after)} | {after['tp']-before['tp']:+d} / {after['fp']-before['fp']:+d} |")
    lines += [""] + confusion_tables(summaries)
    lines += ["| EDF | Jev decisions | GPT reviewed / changed | Mapping new / promoted | API calls | Unavailable gold Narrow |",
              "| --- | --- | --- | --- | --- | --- |"]
    for phenotype, detail in details.items():
        mapping, review = detail['codemapper_run'], detail['gpt_review_run']
        lines.append(f"| {phenotype} | {detail['classification_run']['candidates_classified']} | "
                     f"{review['candidates_reviewed']} / {review['labels_changed']} | "
                     f"{len(mapping['new_codes'])} / {len(mapping['promoted_codes'])} | "
                     f"{len(mapping['calls'])} | "
                     f"{detail['metrics']['evaluation_scope'].get('gold_narrow_excluded_as_unavailable','—')} |")
    lines += ["", "| Stage | TP | TN | FP | FN | Micro recall | Micro precision | Micro F1 | Mean EDF F1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    aggregate = {}
    for stage in ('jev', 'gpt', 'codemapper'):
        rows = [r for r in summaries if r['stage'] == stage]
        tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp','fp','fn'))
        tn = sum(r['tn'] for r in rows)
        recall = tp/(tp+fn) if tp+fn else 0
        precision = tp/(tp+fp) if tp+fp else 0
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        macro_f1 = sum(r['f1'] for r in rows)/len(rows) if rows else 0
        aggregate[stage] = dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, precision=precision, f1=f1,
                                mean_edf_f1=macro_f1)
        lines.append(f"| {stage} | {tp} | {tn} | {fp} | {fn} | {recall:.1%} | {precision:.1%} | {f1:.1%} | {macro_f1:.1%} |")
    reviewed = sum(d['gpt_review_run']['candidates_reviewed'] for d in details.values())
    eligible = sum(d['gpt_review_run']['eligible_candidates'] for d in details.values())
    changed = sum(d['gpt_review_run']['labels_changed'] for d in details.values())
    calls = sum(len(d['codemapper_run']['calls']) for d in details.values())
    hits = sum(d['codemapper_run']['cache_hits'] for d in details.values())
    elapsed = sum(d['codemapper_run']['seconds'] for d in details.values())
    lines += ["", f"GPT reviewed {reviewed} of {eligible} eligible decisions and changed {changed} labels. "
              f"The review cap left {eligible-reviewed} eligible decisions unreviewed. "
              "A low threshold can select few or no candidates: binary-choice confidence "
              "derived from the maximum of normalized label probabilities is at least 0.5.", "",
              f"CodeMapper made {calls} HTTP requests (including sessions), with {hits} cache hits, "
              f"and took {elapsed:.1f} seconds including pacing. Requests were sequential, "
              f"paced at {mapping_config['requests_per_minute']:g} per minute across clients using the same cache; "
              f"batches contained at most {mapping_config['batch_size']} codes/CUIs. "
              "This is our interim limit, not a negotiated service allowance.", ""]
    if details:
        before, after = aggregate['gpt'], aggregate['codemapper']
        lines += [f"CodeMapper changed aggregate recall by {100*(after['recall']-before['recall']):+.1f} "
                  f"percentage points, precision by {100*(after['precision']-before['precision']):+.1f}, "
                  f"and F1 by {100*(after['f1']-before['f1']):+.1f}. "
                  f"It recovered {after['tp']-before['tp']} additional gold Narrow codes and "
                  f"introduced {after['fp']-before['fp']} additional reference-negative codes.", ""]
        if after['f1'] < before['f1']:
            lines += ["In this run, automatic inclusion lowered micro F1. Keep the mapping stage "
                      "optional; review its additions against the EDF before treating it as an improvement.", ""]
        elif after['f1'] > before['f1']:
            lines += ["In this run, automatic inclusion improved micro F1. The single-run result "
                      "supports further evaluation; review mapping additions before adopting it as the default.", ""]
    lines += ["Limitations: one stochastic classification run per EDF; no uncertainty estimates. "
              "Shared CUIs do not guarantee equal clinical specificity or EDF eligibility. "
              "An incomplete reference can count valid additions as false positives. "
              "Vocabulary/version gaps are excluded from the denominator and reported separately. "
              "Read-v2 outputs have no gold coverage in these eight references and are not scored. "
              "Results compare stages within this run, not against a previous run with different retrieval.", "",
              "Artifacts: `manifest.json`, `summary.csv`, `aggregate.json`, and per-EDF "
              "`jev.json/csv`, `gpt.json/csv`, `codemapper.json/csv`, `mapping_changes.csv`, "
              "and `details.json`. Stage CSVs contain all decisions, including Exclude; "
              "select `tags=narrow` for the positive codelist.", ""]
    (directory / 'report.md').write_text('\n'.join(lines))
    (directory / 'aggregate.json').write_text(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/codemapper_experiment.yaml')
    parser.add_argument('--phenotypes', nargs='+')
    parser.add_argument('--output-dir', type=Path, default=Path('results/experiments/codemapper_edfs'))
    parser.add_argument('--report-only', action='store_true', help='Refresh artifacts from completed EDF runs without API calls')
    args = parser.parse_args()
    config = load_config(args.config)
    phenotypes = args.phenotypes or sorted(config.paths.aesi_datasets)
    unknown = set(phenotypes) - config.paths.aesi_datasets.keys()
    if unknown:
        parser.error(f'Unknown EDFs: {sorted(unknown)}')
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
                'phenotypes': phenotypes, 'input_sha256': {}}
    for phenotype in phenotypes:
        for path in (Path(config.paths.edf_dir) / (phenotype + '.yaml'),
                     Path(config.paths.aesi_datasets[phenotype])):
            manifest['input_sha256'][str(path)] = file_hash(path)
    for path in config.paths.code_systems.values():
        manifest['input_sha256'][path] = file_hash(path)
    path = directory / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['config'] != manifest['config'] or previous['input_sha256'] != manifest['input_sha256']:
            parser.error('Existing output belongs to different settings/inputs; use a new output directory')
        manifest = previous
    else:
        path.write_text(json.dumps(manifest, indent=2))
    summaries, details, resources = [], {}, {}
    for phenotype in phenotypes:
        destination = directory / phenotype
        destination.mkdir(exist_ok=True)
        detail_path = destination / 'details.json'
        if detail_path.exists():
            detail = json.loads(detail_path.read_text())
        else:
            if args.report_only:
                parser.error(f'No completed results for {phenotype}')
            config.phenotype = phenotype
            _, gold = load_aesi_dataset(config.paths.aesi_datasets[phenotype])
            config.evaluation.compare_vocabularies = [v for v in config.retrieval.vocabularies
                                                       if v in set(gold['vocabulary'])]
            def progress(message, completed, total):
                suffix = f' {completed}/{total}' if completed is not None else ''
                print(f'[{phenotype}] {message}{suffix}', flush=True)
            def checkpoint(stage, records):
                (destination / f'{stage}.json').write_text(json.dumps(records, indent=2))
                to_predicted_codelist(records).to_csv(destination / f'{stage}.csv', index=False)
            started = perf_counter()
            result = run_pipeline(config, progress_callback=progress, stage_callback=checkpoint,
                                  resource_cache=resources)
            detail = {k: result[k] for k in ('metrics','classification_run','gpt_review_run','codemapper_run')}
            detail['seconds'] = round(perf_counter()-started,3)
            detail_path.write_text(json.dumps(detail, indent=2))
            before = {(c['code'],c['vocabulary']): c for c in result['stage_decisions']['gpt']}
            changes = []
            for c in result['stage_decisions']['codemapper']:
                if c['decision_origin'] == 'codemapper':
                    prior = before.get((c['code'], c['vocabulary']))
                    changes.append({'code':c['code'], 'vocabulary':c['vocabulary'],
                                    'description':c['description'], 'prior_label':prior['label'] if prior else 'Unclassified',
                                    'new_label':c['label'], 'provenance':json.dumps(c['mapping_provenance'])})
            pd.DataFrame(changes, columns=['code','vocabulary','description','prior_label','new_label','provenance']).to_csv(destination/'mapping_changes.csv',index=False)
        # Refresh exports and annotate mappings for reference-based inspection.
        _, gold = load_aesi_dataset(config.paths.aesi_datasets[phenotype])
        scope = detail['metrics']['evaluation_scope']
        selected = set(scope['selected_vocabularies'])
        narrow_total = int(((gold['label'] == 'Narrow') & gold['vocabulary'].isin(selected)).sum())
        narrow_available = detail['metrics']['stages']['jev']['narrow_loss_breakdown']['gold_narrow_available']
        scope.update(gold_narrow_in_selected_vocabularies=narrow_total,
                     gold_narrow_available=narrow_available,
                     gold_narrow_excluded_as_unavailable=narrow_total-narrow_available)
        detail_path.write_text(json.dumps(detail,indent=2))
        gold_by_key = {(standardize_code(r.code,r.vocabulary),r.vocabulary):r.label
                       for r in gold.itertuples(index=False)}
        for stage in ('jev','gpt','codemapper'):
            records = json.loads((destination/f'{stage}.json').read_text())
            to_predicted_codelist(records).to_csv(destination/f'{stage}.csv',index=False)
        changes_path = destination/'mapping_changes.csv'
        changes = pd.read_csv(changes_path,dtype=str,keep_default_na=False)
        changes['gold_label'] = [gold_by_key.get((standardize_code(c,v),v),'Absent')
                                for c,v in zip(changes['code'],changes['vocabulary'])]
        changes['evaluation_status'] = ['Unscored vocabulary' if v not in selected else
                                        'Gold Narrow recovered' if label == 'Narrow' else 'Reference-negative'
                                        for v,label in zip(changes['vocabulary'],changes['gold_label'])]
        changes.to_csv(changes_path,index=False)
        details[phenotype] = detail
        for stage, metrics in detail['metrics']['stages'].items():
            counts = metrics['narrow_codelist']
            summaries.append({'phenotype':phenotype,'stage':stage,'tp':counts['true_positives'],
                              'tn':metrics['classification']['true_negatives'],
                              'fp':counts['false_positives'],'fn':counts['false_negatives'],
                              'recall':counts['sensitivity'],'precision':counts['precision'],'f1':counts['f1'],
                              'gold_unavailable':scope['gold_rows_excluded_as_unavailable'],
                              'gold_narrow_unavailable':scope['gold_narrow_excluded_as_unavailable']})
        pd.DataFrame(summaries).to_csv(directory/'summary.csv',index=False)
        write_report(directory, manifest, summaries, details)
        print(f'[{phenotype}] complete',flush=True)
    manifest.setdefault('finished_utc', datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest,indent=2))
    print(f'Report: {directory / "report.md"}',flush=True)


if __name__ == '__main__':
    main()
