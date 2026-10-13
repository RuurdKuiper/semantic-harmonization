"""Resumable study stages; validation selects settings, tests never select them."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd

from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import preprocess_corpus, standardize_code
from src.llm.classify import ClassifiedCandidate, llm_classify, select_gpt_review_candidates
from src.retrieval.embeddings import EmbeddingIndex, create_embedding_model, embedding_cache_key
from src.retrieval.hybrid import HybridCandidate
from src.retrieval.lexical import build_lexical_index, score_lexical
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ('open_weight', 'proprietary')
STUDY_IMPLEMENTATION = '20261008.1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_manifest():
    packages={}
    for name in ('torch','transformers','openai','laya','llm2jev','mlx-vlm','sentence-transformers'):
        try: packages[name]=version(name)
        except PackageNotFoundError: packages[name]=None
    receipt=ROOT/'.model-cache/model_revisions.json'
    models=json.loads(receipt.read_text()) if receipt.exists() else {}
    return {'packages':packages,'models':{name:info['revision'] for name,info in models.items()}}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    span = np.ptp(values)
    return (values-values.min())/span if span > 1e-12 else np.zeros_like(values)


def ranking_metrics(order, positive_mask, budgets):
    """Average precision and recall at budgets over available Narrow references."""
    hits = np.asarray(positive_mask, dtype=bool)[order]
    positives = int(hits.sum())
    ranks = np.flatnonzero(hits)+1
    result = {'available_narrow': positives,
              'average_precision': float(np.sum(np.arange(1, positives+1)/ranks)/positives) if positives else 0.,
              'last_positive_rank': int(ranks[-1]) if positives else None}
    for k in budgets:
        result[f'recall_at_{k}'] = float(hits[:k].sum()/positives) if positives else 0.
    return result


def confusion_metrics(records, gold, corpus):
    """End-to-end Narrow metrics with assessed TN and corpus-scoped positives."""
    vocabularies = set(gold.vocabulary)
    available = {(standardize_code(r.code,r.vocabulary), r.vocabulary) for r in corpus.itertuples()}
    reference = {(standardize_code(r.code,r.vocabulary), r.vocabulary):r.label for r in gold.itertuples()}
    positives = {key for key,label in reference.items() if label=='Narrow' and key in available}
    decisions = {(standardize_code(r['code'],r['vocabulary']),r['vocabulary']):r['label']
                 for r in records if r['vocabulary'] in vocabularies}
    included = {key for key,label in decisions.items() if label=='Narrow'}
    tp = len(positives & included); fp = len(included-positives); fn = len(positives-included)
    tn = sum(key not in positives and label!='Narrow' for key,label in decisions.items())
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn, 'available_narrow':len(positives),
            'unavailable_narrow':sum(label=='Narrow' and key not in available for key,label in reference.items()),
            'recall':tp/(tp+fn) if tp+fn else 0., 'precision':tp/(tp+fp) if tp+fp else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'fn_unclassified':len(positives-decisions.keys()),
            'fn_classified':sum(key in positives and label!='Narrow' for key,label in decisions.items())}


def choose_winner(rows, required_cases, key='f1'):
    """Equal-case means; reject incomplete competitors before selection."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['name'], []).append(row)
    candidates = []
    for name, group in grouped.items():
        if len(group) != len(required_cases) or {r['phenotype'] for r in group} != set(required_cases):
            raise ValueError(f'Incomplete validation results for {name}.')
        candidates.append((sum(r[key] for r in group)/len(group),name))
    if not candidates:
        raise ValueError('No completed competitors available.')
    # Stable name ordering resolves exact ties without consulting test results.
    score, name = sorted(candidates,key=lambda r:(-r[0],r[1]))[0]
    return {'name':name,'score':score,'metric':f'mean_edf_{key}'}


class Study:
    def __init__(self, config):
        self.config = config
        self.base = load_config(ROOT/config['base_pipeline_config'])
        self.output = ROOT/config['output_dir']; self.output.mkdir(parents=True,exist_ok=True)
        validation, test = config['validation_edfs'], config['test_edfs']
        if set(validation) & set(test) or len(set(validation)) != len(validation):
            raise ValueError('Validation and test EDFs must be distinct and disjoint.')
        self.corpus = None; self.index = None; self.encoder = None; self.lexical_indexes = None
        self.requests = 0
        self.source_signatures={}
        save_json(self.output/'config_snapshots'/f'{fingerprint(config)}.json',config)

    def load_corpus(self):
        if self.corpus is None:
            self.corpus = pd.concat([preprocess_corpus(pd.read_parquet(
                ROOT/self.base.paths.codes_parquet_dir/f'codes_{v}.parquet'))
                for v in self.base.retrieval.vocabularies],ignore_index=True)
        return self.corpus

    def edf(self, phenotype):
        if hasattr(self, 'external_edfs'):
            return self.external_edfs[phenotype]
        return load_edf(phenotype,ROOT/self.base.paths.edf_dir)

    def source_signature(self, phenotype):
        if phenotype in self.source_signatures:return self.source_signatures[phenotype]
        if hasattr(self, 'external_source'):
            result=fingerprint({'source':self.external_source,'edf':asdict(self.edf(phenotype)),
                                'retrieval':self.config['retrieval'],'implementation':STUDY_IMPLEMENTATION})
        else:result=fingerprint({'edf':file_hash(ROOT/self.base.paths.edf_dir/f'{phenotype}.yaml'),
                            'corpora':{v:file_hash(ROOT/self.base.paths.codes_parquet_dir/f'codes_{v}.parquet')
                                       for v in self.base.retrieval.vocabularies},
                            'retrieval':self.config['retrieval'], 'implementation':STUDY_IMPLEMENTATION})
        self.source_signatures[phenotype]=result
        return result

    def scores(self, phenotype):
        corpus = self.load_corpus(); signature = self.source_signature(phenotype)
        cache = self.output/'scores'/f'{phenotype}_{signature}.npz'
        if cache.exists():
            with np.load(cache) as arrays:
                return arrays['lexical'],arrays['embedding']
        retrieval = self.config['retrieval']
        if self.index is None:
            identity = embedding_cache_key(retrieval['embedding_provider'],retrieval['embedding_model'],
                                           retrieval['embedding_dimensions'])
            caches = [ROOT/self.base.paths.embeddings_dir/f'embeddings_{v}_{identity}.npz'
                      for v in self.base.retrieval.vocabularies]
            if not all(p.exists() for p in caches):
                raise RuntimeError('Missing embedding caches. Build approved indexes separately; the study never automatically embeds the vocabulary.')
            self.encoder = create_embedding_model(retrieval['embedding_provider'],retrieval['embedding_model'],
                                                  dimensions=retrieval['embedding_dimensions'])
            self.index = EmbeddingIndex.load_for_vocabularies(corpus,self.base.retrieval.vocabularies,
                ROOT/self.base.paths.embeddings_dir,model_name=retrieval['embedding_model'],model=self.encoder,
                provider=retrieval['embedding_provider'],dimensions=retrieval['embedding_dimensions'])
            assert list(zip(self.index.codes,self.index.vocabularies)) == list(zip(corpus.code,corpus.vocabulary))
            self.lexical_indexes = {v:build_lexical_index(corpus[corpus.vocabulary==v]) for v in self.base.retrieval.vocabularies}
        edf = self.edf(phenotype)
        lexical_query=edf.preferred_name if hasattr(self,'external_source') else edf.to_lexical_query()
        embedding_query=edf.preferred_name if hasattr(self,'external_source') else edf.to_embedding_query()
        lexical = np.zeros(len(corpus))
        for vocabulary,index in self.lexical_indexes.items():
            mask = corpus.vocabulary==vocabulary
            lexical[mask] = score_lexical(lexical_query,index,int(mask.sum()))
        embedding = self.index.score_all(embedding_query,self.encoder)
        lexical,embedding = normalize(lexical),normalize(embedding)
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,lexical=lexical,embedding=embedding)
        return lexical,embedding

    def ordered_candidates(self, phenotype, lexical_weight):
        corpus = self.load_corpus(); lexical,embedding = self.scores(phenotype)
        scores = np.round(lexical_weight*lexical+(1-lexical_weight)*embedding,6)
        order = np.argsort(-scores,kind='stable')
        # Construct records lazily in batches instead of 620,000 dataclasses.
        return order, scores, lexical, embedding

    def retrieval(self):
        rows = []; corpus = self.load_corpus()
        for phenotype in self.config['validation_edfs']:
            _,gold = load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
            positive = {(standardize_code(r.code,r.vocabulary),r.vocabulary)
                        for r in gold.itertuples() if r.label=='Narrow'}
            mask = np.array([(standardize_code(r.code,r.vocabulary),r.vocabulary) in positive
                             for r in corpus.itertuples()])
            lexical,embedding = self.scores(phenotype)
            for weight in self.config['retrieval']['lexical_weights']:
                order = np.argsort(-np.round(weight*lexical+(1-weight)*embedding,6),kind='stable')
                row = {'phenotype':phenotype,'lexical_weight':weight,'embedding_weight':1-weight,
                       **ranking_metrics(order,mask,self.config['retrieval']['recall_budgets'])}
                rows.append(row)
            print(f'[retrieval] {phenotype}: {int(mask.sum())} available Narrow references',flush=True)
        pd.DataFrame(rows).to_csv(self.output/'retrieval.csv',index=False)
        means = pd.DataFrame(rows).groupby('lexical_weight',as_index=False).average_precision.mean()
        winner = means.sort_values(['average_precision','lexical_weight'],ascending=[False,True]).iloc[0]
        selection = {'lexical_weight':float(winner.lexical_weight),'embedding_weight':float(1-winner.lexical_weight),
                     'mean_average_precision':float(winner.average_precision),'validation_edfs':self.config['validation_edfs'],
                     'retrieval':self.config['retrieval'],'config_hash':fingerprint(self.config)}
        save_json(self.output/'retrieval_selection.json',selection)
        print('Retrieval selection:',json.dumps(selection),flush=True)
        return selection

    def _request(self):
        if self.requests >= self.config['execution']['max_requests']:
            raise RuntimeError('Operational request ceiling reached; completed batches remain resumable.')
        self.requests += 1

    def classify(self, phenotype, competitor, weight):
        corpus = self.load_corpus(); edf = self.edf(phenotype)
        order,scores,lexical,embedding = self.ordered_candidates(phenotype,weight)
        settings = self.config['classification']; size=settings['batch_size']
        limit=min(len(order),settings.get('max_candidates') or len(order))
        signature = fingerprint({'source':self.source_signature(phenotype),'competitor':competitor,
                                 'weight':weight,'stopping':settings})
        path = self.output/'decisions'/competitor['name']/f'{phenotype}.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous and previous['signature'] != signature:
            raise ValueError(f'Changed decision inputs for {path}; use a new output directory.')
        if previous.get('complete'): return previous
        records = previous.get('records',[]); counts=previous.get('narrow_counts',[])
        timings=previous.get('batch_seconds',[])
        for start in range(len(records),limit,size):
            indexes=order[start:min(start+size,limit)]
            batch=[HybridCandidate(rank=start+i+1,code=corpus.iloc[idx].code,
                description=corpus.iloc[idx].description,vocabulary=corpus.iloc[idx].vocabulary,
                lexical_score=float(lexical[idx]),embedding_score=float(embedding[idx]),score=float(scores[idx]))
                for i,idx in enumerate(indexes)]
            self._request()
            print(f"[decisions/{competitor['name']}] {phenotype}: processing {start+1}--{start+len(batch)}",flush=True)
            started=perf_counter()
            results=llm_classify(batch,edf,provider=competitor['provider'],model=competitor['model'],
                                 batch_size=size,max_retries=3,use_possible_category=False)
            records += [asdict(item) for item in results]
            counts.append(sum(item.label=='Narrow' for item in results))
            timings.append(perf_counter()-started)
            last=counts[-settings['consecutive_sparse_batches']:]
            stopped = (len(counts)>=settings['minimum_batches'] and
                       len(last)==settings['consecutive_sparse_batches'] and
                       all(n<=settings['sparse_narrow_threshold'] for n in last))
            previous={'signature':signature,'records':records,'narrow_counts':counts,
                      'complete':stopped or len(records)==limit,'stopped_early':stopped,
                      'candidate_ceiling_hit':not stopped and len(records)==limit and limit<len(order),
                      'stop_reason':'adaptive_zero_narrow' if stopped else
                                    'candidate_ceiling' if len(records)==limit and limit<len(order) else
                                    'corpus_exhausted' if len(records)==len(order) else 'in_progress',
                      'competitor':competitor,'lexical_weight':weight,'batch_seconds':timings,
                      'model_seconds':sum(timings),'runtime':runtime_manifest()}
            save_json(path,previous)
            print(f"[decisions/{competitor['name']}] {phenotype}: {len(records)} classified",flush=True)
            if stopped: break
        return previous

    def decisions(self, names=None):
        weight=json.loads((self.output/'retrieval_selection.json').read_text())['lexical_weight']
        rows=[]
        for competitor in self.config['decision_models']:
            if names and competitor['name'] not in names:continue
            rows=[]
            from src.llm.model_paths import release_local_models
            release_local_models()
            for phenotype in self.config['validation_edfs']:
                result=self.classify(phenotype,competitor,weight)
                _,gold=load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
                rows.append({'name':competitor['name'],'branch':competitor['branch'],'phenotype':phenotype,
                             'classified':len(result['records']),'model_seconds':result.get('model_seconds'),
                             'candidate_ceiling_hit':result.get('candidate_ceiling_hit',False),
                             'stop_reason':result.get('stop_reason'),
                             **confusion_metrics(result['records'],gold,self.load_corpus())})
                pd.DataFrame(rows).to_csv(self.output/f"decision_metrics.{competitor['name']}.csv",index=False)
        rows=[]
        for competitor in self.config['decision_models']:
            for phenotype in self.config['validation_edfs']:
                path=self.output/'decisions'/competitor['name']/f'{phenotype}.json'
                if not path.exists():continue
                result=json.loads(path.read_text())
                if not result.get('complete'):continue
                _,gold=load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
                rows.append({'name':competitor['name'],'branch':competitor['branch'],'phenotype':phenotype,
                             'classified':len(result['records']),'model_seconds':result.get('model_seconds'),
                             'candidate_ceiling_hit':result.get('candidate_ceiling_hit',False),
                             'stop_reason':result.get('stop_reason'),
                             **confusion_metrics(result['records'],gold,self.load_corpus())})
        pd.DataFrame(rows).to_csv(self.output/'decision_metrics.partial.csv',index=False)
        if len(rows)!=len(self.config['decision_models'])*len(self.config['validation_edfs']):
            print('Decision comparison remains incomplete; no winners selected.',flush=True)
            return rows
        pd.DataFrame(rows).to_csv(self.output/'decision_metrics.csv',index=False)
        selections={branch:choose_winner([r for r in rows if r['branch']==branch],self.config['validation_edfs'])
                    for branch in BRANCHES}
        selections['overall']=choose_winner(rows,self.config['validation_edfs'])
        save_json(self.output/'decision_selection.json',selections)
        return selections

    def review_case(self, phenotype, competitor, decision):
        settings=self.config['review']; edf=self.edf(phenotype)
        originals=[ClassifiedCandidate(**r) for r in decision['records']]
        selected=select_gpt_review_candidates(originals,confidence_threshold=settings['confidence_threshold'],
                                             max_candidates=settings['max_candidates'])
        path=self.output/'reviews'/competitor['name']/decision['competitor']['name']/f'{phenotype}.json'
        signature_settings={k:v for k,v in settings.items() if k!='qwen_thinking_budget' or competitor['provider']=='mlx'}
        signature=fingerprint({'decision':decision['signature'],'competitor':competitor,'review':signature_settings})
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous and previous['signature']!=signature:
            raise ValueError(f'Changed review inputs for {path}; use a new output directory.')
        if previous.get('complete'):return previous
        replacements=previous.get('replacements',[])
        timings=previous.get('batch_seconds',[])
        for start in range(len(replacements),len(selected),settings['batch_size']):
            batch=selected[start:start+settings['batch_size']]; self._request()
            print(f"[review/{competitor['name']}] {phenotype}: processing {start+1}--{start+len(batch)} of {len(selected)}",flush=True)
            started=perf_counter()
            reviewed=llm_classify(batch,edf,provider=competitor['provider'],model=competitor['model'],
                                  batch_size=settings['batch_size'],reasoning_effort=settings['reasoning_effort'],
                                  thinking_budget=settings.get('qwen_thinking_budget') if competitor['provider']=='mlx' else None)
            replacements += [asdict(r) for r in reviewed]
            timings.append(perf_counter()-started)
            save_json(path,{'signature':signature,'replacements':replacements,'complete':False,'batch_seconds':timings})
        by_key={(r['code'],r['vocabulary']):r for r in replacements}
        merged=[]
        for original in originals:
            record=asdict(original); replacement=by_key.get((original.code,original.vocabulary))
            if replacement:
                record.update(label=replacement['label'],confidence=replacement['confidence'],
                    explanation=replacement['explanation'],decision_origin='review',
                    prior_label=original.label,prior_confidence=original.confidence)
            merged.append(record)
        result={'signature':signature,'complete':True,'replacements':replacements,'records':merged,
                'reviewed':len(selected),'eligible':sum(c.confidence<settings['confidence_threshold'] for c in originals),
                'labels_changed':sum(r['label']!=o.label for r,o in zip(merged,originals)),
                'reviewer':competitor,'decision_competitor':decision['competitor'],
                'batch_seconds':timings,'model_seconds':sum(timings),'runtime':runtime_manifest()}
        save_json(path,result); return result

    def review(self):
        selections=json.loads((self.output/'decision_selection.json').read_text());rows=[]
        # All reviewers compare the same best overall preliminary answers.
        # Branch-specific final configurations also use compatible decision models.
        for competitor in self.config['review_models']:
            from src.llm.model_paths import release_local_models
            release_local_models()
            for target in ('overall',competitor['branch']):
                name=selections[target]['name']
                for phenotype in self.config['validation_edfs']:
                    decision=json.loads((self.output/'decisions'/name/f'{phenotype}.json').read_text())
                    print(f"[review/{competitor['name']}/{target}] {phenotype}",flush=True)
                    result=self.review_case(phenotype,competitor,decision)
                    _,gold=load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
                    rows.append({'name':competitor['name'],'branch':target,'phenotype':phenotype,
                                 'decision_name':name,'reviewed':result['reviewed'],'eligible':result['eligible'],
                                 **confusion_metrics(result['records'],gold,self.load_corpus())})
                    pd.DataFrame(rows).to_csv(self.output/'review_metrics.partial.csv',index=False)
        pd.DataFrame(rows).to_csv(self.output/'review_metrics.csv',index=False)
        chosen={branch:choose_winner([r for r in rows if r['branch']==branch],self.config['validation_edfs'])
                for branch in (*BRANCHES,'overall') if any(r['branch']==branch for r in rows)}
        save_json(self.output/'review_selection.json',chosen);return chosen

    def mapping_case(self, branch, phenotype, review):
        from dataclasses import replace
        from src.mapping.codemapper import expand_narrow_matches
        path=self.output/'mapping'/branch/f'{phenotype}.json'
        settings=replace(self.base.codemapper,enabled=True,
                         batch_size=self.config['mapping']['batch_size'],
                         requests_per_minute=self.config['mapping']['requests_per_minute'])
        signature=fingerprint({'review':review['signature'],'mapping':asdict(settings)})
        if path.exists():
            result=json.loads(path.read_text())
            if result['signature']!=signature:raise ValueError(f'Changed mapping inputs: {path}')
            return result
        print(f'[mapping/{branch}] {phenotype}: direct CUI matches, six requests/minute',flush=True)
        records,info=expand_narrow_matches([ClassifiedCandidate(**r) for r in review['records']],
            self.load_corpus(),self.base.retrieval.vocabularies,settings)
        result={'signature':signature,'records':[asdict(r) for r in records],'mapping_run':info}
        save_json(path,result);return result

    def mapping(self):
        decision=json.loads((self.output/'decision_selection.json').read_text())
        review=json.loads((self.output/'review_selection.json').read_text());rows=[]
        for branch in BRANCHES:
            if branch not in review:raise ValueError(f'Missing completed {branch} reviewer comparison.')
            for phenotype in self.config['validation_edfs']:
                baseline=json.loads((self.output/'reviews'/review[branch]['name']/decision[branch]['name']/f'{phenotype}.json').read_text())
                mapped=self.mapping_case(branch,phenotype,baseline)
                _,gold=load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
                for stage,result in (('review',baseline),('codemapper',mapped)):
                    rows.append({'branch':branch,'phenotype':phenotype,'stage':stage,
                                 **confusion_metrics(result['records'],gold,self.load_corpus())})
                pd.DataFrame(rows).to_csv(self.output/'mapping_metrics.partial.csv',index=False)
        pd.DataFrame(rows).to_csv(self.output/'mapping_metrics.csv',index=False)
        selections={}
        for branch in BRANCHES:
            before=np.mean([r['f1'] for r in rows if r['branch']==branch and r['stage']=='review'])
            after=np.mean([r['f1'] for r in rows if r['branch']==branch and r['stage']=='codemapper'])
            selections[branch]={'enabled':bool(after>before),'review_mean_f1':float(before),
                                'mapping_mean_f1':float(after),'selection_data':'validation only'}
        save_json(self.output/'mapping_selection.json',selections);return selections

    def freeze(self):
        if self.config.get('pending_reviewers'):raise ValueError('Reviewer choices remain unspecified.')
        decision=json.loads((self.output/'decision_selection.json').read_text())
        review=json.loads((self.output/'review_selection.json').read_text())
        mapping=json.loads((self.output/'mapping_selection.json').read_text())
        branches={}
        for branch in BRANCHES:
            branches[branch]={
                'decision':next(m for m in self.config['decision_models'] if m['name']==decision[branch]['name']),
                'reviewer':next(m for m in self.config['review_models'] if m['name']==review[branch]['name']),
                'mapping':mapping[branch],
                'offline_capable':branch=='open_weight' and not mapping[branch]['enabled']}
        sources={}
        for vocabulary in self.base.retrieval.vocabularies:
            path=ROOT/self.base.paths.codes_parquet_dir/f'codes_{vocabulary}.parquet'
            sources[str(path.relative_to(ROOT))]=file_hash(path)
        path=ROOT/self.config['base_pipeline_config']
        sources[str(path.relative_to(ROOT))]=file_hash(path)
        for phenotype in self.config['validation_edfs']:
            for path in (ROOT/self.base.paths.edf_dir/f'{phenotype}.yaml',ROOT/self.base.paths.aesi_datasets[phenotype]):
                sources[str(path.relative_to(ROOT))]=file_hash(path)
        frozen={'frozen_at':datetime.now(timezone.utc).isoformat(),'validation_edfs':self.config['validation_edfs'],
                'test_edfs':self.config['test_edfs'],'test_status':self.config['test_status'],
                'retrieval_selection':json.loads((self.output/'retrieval_selection.json').read_text()),
                'classification':self.config['classification'],'review':self.config['review'],
                'mapping_settings':self.config['mapping'],'benchmark':self.config['benchmark'],
                'branches':branches,'validation_source_hashes':sources,'config_hash':fingerprint(self.config),
                'runtime':runtime_manifest(),'implementation':STUDY_IMPLEMENTATION,
                'identifier_adapter_sha256':file_hash(Path(__file__).with_name('cprd_mapping.py')),
                'implementation_sha256':file_hash(__file__)}
        path=self.output/'frozen_pipelines.json'
        if path.exists():raise ValueError('Pipelines are already frozen. Use a new output directory to select new settings.')
        save_json(path,frozen);(self.output/'frozen_pipelines.sha256').write_text(file_hash(path))
        return frozen

    def frozen(self):
        path=self.output/'frozen_pipelines.json'
        assert file_hash(path)==(self.output/'frozen_pipelines.sha256').read_text()
        result=json.loads(path.read_text())
        if result['implementation_sha256']!=file_hash(__file__):
            raise ValueError('Study implementation changed after freezing; start a separate experiment.')
        if result.get('identifier_adapter_sha256')!=file_hash(Path(__file__).with_name('cprd_mapping.py')):
            raise ValueError('Benchmark identifier adapter changed after freezing; start a separate experiment.')
        for path,expected in result['validation_source_hashes'].items():
            if file_hash(ROOT/path)!=expected:raise ValueError(f'Validation source changed after freezing: {path}')
        # Later phases consume frozen settings, regardless of edited study YAML.
        self.config={**self.config,'classification':result['classification'],'review':result['review'],
                     'retrieval':result['retrieval_selection']['retrieval'],
                     'mapping':result['mapping_settings'],'benchmark':result['benchmark']}
        return result

    def test(self):
        frozen=self.frozen();rows=[];weight=frozen['retrieval_selection']['lexical_weight']
        for branch,settings in frozen['branches'].items():
            for phenotype in frozen['test_edfs']:
                decision=self.classify(phenotype,settings['decision'],weight)
                review=self.review_case(phenotype,settings['reviewer'],decision)
                final=self.mapping_case(branch,phenotype,review) if settings['mapping']['enabled'] else review
                _,gold=load_aesi_dataset(ROOT/self.base.paths.aesi_datasets[phenotype])
                for stage,result in (('decision',decision),('review',review),('final',final)):
                    rows.append({'branch':branch,'phenotype':phenotype,'stage':stage,
                                 **confusion_metrics(result['records'],gold,self.load_corpus())})
                pd.DataFrame(rows).to_csv(self.output/'test_metrics.csv',index=False)
        return rows

    def benchmark_mapping_case(self,branch,phenotype,review,aliases,available,coverage):
        from dataclasses import replace
        from experiments.pipeline_design.cprd_mapping import expand_cprd_matches
        settings=replace(self.base.codemapper,enabled=True,
            batch_size=self.config['mapping']['batch_size'],
            requests_per_minute=self.config['mapping']['requests_per_minute'])
        signature=fingerprint({'review':review['signature'],'browser':self.external_source,
            'adapter_sha256':file_hash(Path(__file__).with_name('cprd_mapping.py')),'mapping':asdict(settings)})
        path=self.output/'mapping'/branch/f'{phenotype}.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['signature']!=signature:raise ValueError(f'Changed benchmark mapping inputs: {path}')
            return result
        print(f'[benchmark/mapping/{branch}] {phenotype}: direct CUI links through validated browser identifiers',flush=True)
        records,info=expand_cprd_matches([ClassifiedCandidate(**r) for r in review['records']],
                                        self.corpus,aliases,available,settings)
        result={'signature':signature,'records':[asdict(r) for r in records],
                'mapping_run':info,'identifier_coverage':coverage}
        save_json(path,result);return result

    def benchmark(self):
        import csv
        from copy import deepcopy
        from experiments.matthewman_2026.experiment import (PHENOTYPES,load_cprd_browser,
            load_reference_codelists,scope_reference_to_corpus,build_classification_edf,
            score_whole_codelist,score_as_dict)
        frozen=self.frozen();settings=frozen['benchmark'];rows=[]
        full,read=load_cprd_browser(ROOT/settings['browser'])
        references=load_reference_codelists(ROOT/settings['references'])
        raw_browser=None
        if any(models['mapping']['enabled'] for models in frozen['branches'].values()):
            path=ROOT/settings['browser']
            if path.suffix.lower()=='.dta':raw_browser=pd.read_stata(path,convert_categoricals=False)
            else:
                with path.open(encoding='utf-8-sig') as handle:
                    delimiter=csv.Sniffer().sniff(handle.readline(),delimiters=',\t;|').delimiter
                raw_browser=pd.read_csv(path,sep=delimiter,dtype=str,keep_default_na=False,encoding='utf-8-sig')
        retrieval=self.config['retrieval']
        encoder=create_embedding_model(retrieval['embedding_provider'],retrieval['embedding_model'],
                                       dimensions=retrieval['embedding_dimensions'])
        # CPRD local indexes may be built on-device; no descriptions go to an embedding API.
        if retrieval['embedding_provider']!='local':
            raise ValueError('Build separately approved CPRD embedding caches before using a non-local benchmark encoder.')
        identity=embedding_cache_key(retrieval['embedding_provider'],retrieval['embedding_model'],
                                     retrieval['embedding_dimensions'])
        index=EmbeddingIndex.from_cache_or_build(full,self.output/'benchmark'/f'embeddings_full_{identity}.npz',
            model_name=retrieval['embedding_model'],model=encoder,load_model=False,show_progress_bar=True)
        full_positions={code:i for i,code in enumerate(index.codes)}
        for subset in settings['subsets']:
            corpus=full if subset=='full' else read
            positions=[full_positions[code] for code in corpus.code]
            cfg=deepcopy(self.config);cfg['output_dir']=str((self.output/'benchmark'/subset).relative_to(ROOT))
            worker=Study(cfg);worker.corpus=corpus;worker.encoder=encoder
            worker.index=EmbeddingIndex(model=None,codes=corpus.code.tolist(),descriptions=corpus.description.tolist(),
                vocabularies=corpus.vocabulary.tolist(),embeddings=index.embeddings[positions])
            worker.lexical_indexes={corpus.vocabulary.iloc[0]:build_lexical_index(corpus)}
            worker.external_edfs={name:build_classification_edf(name,preferred,settings['policy'])
                                  for name,(preferred,_) in PHENOTYPES.items()}
            worker.external_source={'browser':file_hash(ROOT/settings['browser']),'subset':subset}
            if raw_browser is not None:
                from experiments.pipeline_design.cprd_mapping import browser_aliases
                native=pd.concat([pd.read_parquet(ROOT/self.base.paths.codes_parquet_dir/f'codes_{voc}.parquet')
                                  for voc in ('RCD2','SNOMEDCT_US')],ignore_index=True)
                aliases,available,coverage=browser_aliases(raw_browser,corpus,native)
                save_json(worker.output/'identifier_coverage.json',coverage)
            for branch,models in frozen['branches'].items():
                from src.llm.model_paths import release_local_models
                release_local_models()
                for phenotype in PHENOTYPES:
                    print(f'[benchmark/{branch}/{subset}] {phenotype}',flush=True)
                    decision=worker.classify(phenotype,models['decision'],frozen['retrieval_selection']['lexical_weight'])
                    review=worker.review_case(phenotype,models['reviewer'],decision)
                    final=worker.benchmark_mapping_case(branch,phenotype,review,aliases,available,coverage) if models['mapping']['enabled'] else review
                    reference=scope_reference_to_corpus(references[phenotype],corpus)
                    predicted=[r['code'] for r in final['records'] if r['label']=='Narrow']
                    score=score_as_dict(score_whole_codelist(reference,corpus.code.tolist(),predicted,
                                                           [r['code'] for r in decision['records']]))
                    rows.append({'branch':branch,'subset':subset,'phenotype':phenotype,**score})
                    pd.DataFrame(rows).to_csv(self.output/'benchmark_metrics.csv',index=False)
        return rows
