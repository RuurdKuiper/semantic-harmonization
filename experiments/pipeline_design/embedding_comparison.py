#!/usr/bin/env python3
"""Controlled validation-only embedding comparison; never reselect frozen pipelines."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from experiments.pipeline_design import study as study_module
from experiments.pipeline_design.study import Study,file_hash,fingerprint,normalize,ranking_metrics,confusion_metrics
from experiments.pipeline_design.embedding_cache import atomic_npz,build_cache,cached_queries,cosine_scores
from experiments.pipeline_design.parallel_benchmark import atomic_json,locked,redact
from src.data.loaders import load_aesi_dataset
from src.data.preprocessing import standardize_code


class Comparison:
    def __init__(self,config):
        self.config=config;self.output=ROOT/config['output_dir'];self.output.mkdir(parents=True,exist_ok=True)
        self.source_config=yaml.safe_load((ROOT/config['source_study_config']).read_text())
        self.source_output=ROOT/self.source_config['output_dir']
        path=self.source_output/'frozen_pipelines.json';self.frozen=json.loads(path.read_text())
        if file_hash(path)!=(self.source_output/'frozen_pipelines.sha256').read_text():raise ValueError('Frozen-pipeline checksum failed.')
        if file_hash(study_module.__file__)!=self.frozen['implementation_sha256']:raise ValueError('Frozen scientific implementation changed.')
        self.cases=self.source_config['validation_edfs']
        if len(self.cases)!=7 or set(self.cases)&set(self.frozen['test_edfs']):raise ValueError('Comparison must use seven validation EDFs only.')
        for name,expected in self.frozen['validation_source_hashes'].items():
            if file_hash(ROOT/name)!=expected:raise ValueError(f'Frozen validation input changed: {name}')
        protocol={'config':config,'source_config':self.source_config,'frozen_sha256':file_hash(path),
                  'validation_edfs':self.cases,'models':self.frozen['branches']['proprietary'],
                  'classification':self.frozen['classification'],'review':self.frozen['review'],
                  'retrieval':self.frozen['retrieval_selection']['retrieval'],
                  'source_hashes':self.frozen['validation_source_hashes'],
                  'study_implementation_sha256':self.frozen['implementation_sha256'],
                  'scope':'Validation-only study extension; original frozen settings and evaluations remain unchanged.'}
        manifest=self.output/'protocol.json'
        if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Comparison protocol changed; use a new output directory.')
        atomic_json(manifest,protocol)
        atomic_json(self.output/'runner.json',{'started_at':datetime.now(timezone.utc).isoformat(),
            'runner_sha256':file_hash(__file__),'cache_implementation_sha256':file_hash(Path(__file__).with_name('embedding_cache.py')),
            'pid':os.getpid()})
        self.local=self.worker('local',self.output/'rankings/local')
        print('[comparison] Loading the same six-vocabulary validation corpus',flush=True)
        corpus_cache=self.output/'corpus.parquet'
        self.corpus=pd.read_parquet(corpus_cache) if corpus_cache.exists() else self.local.load_corpus()
        if not corpus_cache.exists():self.corpus.to_parquet(corpus_cache,index=False)
        self.local.corpus=self.corpus
        self.remote=self.worker('openai',self.output/'rankings/openai');self.remote.corpus=self.corpus
        self.workers={'local':self.local,'openai':self.remote}
        self.vocabularies=list(self.local.base.retrieval.vocabularies)
        self.folders=[ROOT/config['embedding_cache_dir']/v for v in self.vocabularies]
        print(f'[comparison] {len(self.corpus):,} corpus entries; {len(self.cases)} validation EDFs; test cases excluded',flush=True)

    def worker(self,provider,output):
        cfg=deepcopy(self.source_config);cfg['output_dir']=str(output.relative_to(ROOT))
        cfg['classification']=self.frozen['classification'];cfg['review']=self.frozen['review']
        cfg['retrieval']=deepcopy(self.frozen['retrieval_selection']['retrieval'])
        if provider=='openai':
            cfg['retrieval'].update(embedding_provider='openai',embedding_model=self.config['embedding']['model'],
                                    embedding_dimensions=self.config['embedding']['dimensions'])
        return Study(cfg)

    def status(self,phase,status='running',**details):
        atomic_json(self.output/'status.json',{'phase':phase,'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),
                                              'pid':os.getpid(),**details})

    def local_scores(self):
        for name in self.cases:
            signature=self.local.source_signature(name)
            source=self.source_output/'scores'/f'{name}_{signature}.npz'
            if not source.exists():raise ValueError(f'Verified local scores missing: {source}')
            target=self.local.output/'scores'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
        print('[comparison] Reused all seven verified MiniLM/BM25 score caches; no local model loaded',flush=True)

    def embeddings(self):
        import openai
        if not os.getenv('OPENAI_API_KEY'):raise ValueError('OPENAI_API_KEY is missing from the environment/.env.')
        client=openai.OpenAI(max_retries=2,timeout=120.)
        completed_before=0;total=len(self.corpus)
        for vocabulary,folder in zip(self.vocabularies,self.folders):
            frame=self.corpus[self.corpus.vocabulary==vocabulary].reset_index(drop=True)
            def progress(state):
                completed=completed_before+state['completed_rows']
                print(f'[embeddings total] {completed:,}/{total:,} ({100*completed/total:.1f}%) saved; '
                      f'{vocabulary} tokens={state["total_tokens"]:,}',flush=True)
                self.status('embeddings',vocabulary=vocabulary,completed_rows=completed,total_rows=total)
            state=build_cache(frame,folder,self.config['embedding'],client,progress)
            completed_before+=state['completed_rows']
        queries={name:self.remote.edf(name).to_embedding_query() for name in self.cases}
        vectors=cached_queries(queries,self.output/'queries',self.config['embedding'],client)
        missing=[name for name in self.cases if not (self.remote.output/'scores'/f'{name}_{self.remote.source_signature(name)}.npz').exists()]
        if missing:
            self.status('scoring');print('[comparison] Scoring seven EDF queries against disk-backed OpenAI vectors',flush=True)
            raw=cosine_scores(self.folders,vectors)
            if len(raw)!=len(self.corpus):raise ValueError('Embedding scores do not match corpus order.')
            for j,name in enumerate(self.cases):
                lexical,_=self.local.scores(name)
                path=self.remote.output/'scores'/f'{name}_{self.remote.source_signature(name)}.npz'
                atomic_npz(path,lexical=lexical,embedding=normalize(raw[:,j]))
        usage=[json.loads((folder/'state.json').read_text()) for folder in self.folders]
        atomic_json(self.output/'embedding_usage.json',{'corpus_rows':total,'corpus_tokens':sum(x['total_tokens'] for x in usage),
            'corpus_requests':sum(x['requests'] for x in usage),'corpus_api_seconds':sum(x['seconds'] for x in usage),
            'model':self.config['embedding'],'query_usage_files':[str(p.relative_to(ROOT)) for p in sorted((self.output/'queries').glob('*.json'))],
            'note':'Usage covers successfully checkpointed embedding responses; retries before a checkpoint can incur additional usage.'})

    def retrieval(self,providers=('local','openai')):
        path=self.output/'retrieval.csv';rows=[]
        if path.exists():rows=pd.read_csv(path).query('embedder not in @providers').to_dict('records')
        for provider in providers:
            worker=self.workers[provider]
            for name in self.cases:
                _,gold=load_aesi_dataset(ROOT/worker.base.paths.aesi_datasets[name])
                positives={(standardize_code(r.code,r.vocabulary),r.vocabulary) for r in gold.itertuples() if r.label=='Narrow'}
                mask=np.array([(code,vocabulary) in positives for code,vocabulary in self.corpus[['code','vocabulary']].itertuples(index=False,name=None)])
                lexical,embedding=worker.scores(name)
                for weight in worker.config['retrieval']['lexical_weights']:
                    order=np.argsort(-np.round(weight*lexical+(1-weight)*embedding,6),kind='stable')
                    rows.append({'embedder':provider,'phenotype':name,'lexical_weight':weight,
                                 **ranking_metrics(order,mask,self.config['recall_budgets'])})
                pd.DataFrame(rows).to_csv(path,index=False)
                print(f'[retrieval/{provider}] {name}: all 21 weights scored',flush=True)
        frame=pd.DataFrame(rows);selected={}
        for provider,group in frame.groupby('embedder'):
            means=group.groupby('lexical_weight').average_precision.mean()
            counts=group.groupby('lexical_weight').phenotype.nunique()
            if not (counts==len(self.cases)).all():raise ValueError('Incomplete retrieval grid cannot select a weight.')
            weight=min(means.index,key=lambda w:(-means.loc[w],w))
            selected[provider]={'lexical_weight':float(weight),'mean_average_precision':float(means.loc[weight])}
        atomic_json(self.output/'retrieval_selection.json',selected)
        return selected

    def pipeline(self):
        choices=json.loads((self.output/'retrieval_selection.json').read_text())
        metrics_path=self.output/'pipeline_metrics.csv'
        rows=pd.read_csv(metrics_path).to_dict('records') if metrics_path.exists() else []
        models=self.frozen['branches']['proprietary'];decision_model=models['decision'];reviewer=models['reviewer']
        for provider in ('local','openai'):
            weights=sorted({self.config['fixed_lexical_weight'],choices[provider]['lexical_weight']})
            for weight in weights:
                folder=self.output/'pipelines'/f'{provider}_w{weight:.2f}';worker=self.worker(provider,folder);worker.corpus=self.corpus
                for name in self.cases:
                    signature=worker.source_signature(name)
                    source=self.workers[provider].output/'scores'/f'{name}_{signature}.npz'
                    target=folder/'scores'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():shutil.copyfile(source,target)
                    if provider=='local' and weight==self.frozen['retrieval_selection']['lexical_weight']:
                        self.reuse_baseline(worker,name,weight,decision_model,reviewer)
                    self.status('pipeline',embedder=provider,lexical_weight=weight,phenotype=name)
                    print(f'[pipeline/{provider}/w={weight:.2f}] {name}: Jev then GPT review',flush=True)
                    decision=worker.classify(name,decision_model,weight);review=worker.review_case(name,reviewer,decision)
                    _,gold=load_aesi_dataset(ROOT/worker.base.paths.aesi_datasets[name])
                    for stage,result in (('decision',decision),('review',review)):
                        rows.append({'embedder':provider,'lexical_weight':weight,'phenotype':name,'stage':stage,
                            'classified':len(decision['records']),'reviewed':review['reviewed'],
                            'stop_reason':decision.get('stop_reason'),'candidate_ceiling_hit':decision.get('candidate_ceiling_hit',False),
                            **confusion_metrics(result['records'],gold,self.corpus)})
                    pd.DataFrame(rows).drop_duplicates(['embedder','lexical_weight','phenotype','stage'],keep='last').to_csv(metrics_path,index=False)
                    self.report()

    def reuse_baseline(self,worker,name,weight,decision_model,reviewer):
        expected=fingerprint({'source':worker.source_signature(name),'competitor':decision_model,
                              'weight':weight,'stopping':worker.config['classification']})
        relative=Path('decisions')/decision_model['name']/f'{name}.json'
        original=json.loads((self.source_output/relative).read_text())
        if original['signature']!=expected or not original['complete']:raise ValueError('Baseline decision cannot be reused exactly.')
        review_settings={k:v for k,v in worker.config['review'].items() if k!='qwen_thinking_budget'}
        expected_review=fingerprint({'decision':expected,'competitor':reviewer,'review':review_settings})
        relative_review=Path('reviews')/reviewer['name']/decision_model['name']/f'{name}.json'
        original_review=json.loads((self.source_output/relative_review).read_text())
        if original_review['signature']!=expected_review or not original_review['complete']:raise ValueError('Baseline review cannot be reused exactly.')
        for relative in (relative,relative_review):
            target=worker.output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(self.source_output/relative,target)
        print(f'[reuse] {name}: identical saved MiniLM decisions and reviews; no repeated model calls',flush=True)

    def report(self):
        from experiments.pipeline_design.embedding_comparison_report import write_report
        write_report(self.output,self.config,self.cases)

    def run(self):
        self.status('local_retrieval');self.local_scores();self.retrieval(('local',));self.report()
        self.status('embeddings');self.embeddings()
        self.status('retrieval');self.retrieval();self.report()
        self.pipeline();self.status('complete','complete');self.report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_suffix('.yaml')))
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();config_path=Path(args.config).resolve();config=yaml.safe_load(config_path.read_text())
    output=ROOT/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    load_dotenv(ROOT/'.env')
    if args.detach:
        with locked(output/'launch.lock'):
            with locked(output/'worker.lock',blocking=False):pass
            marker=output/'worker.json'
            if marker.exists():
                try:os.kill(json.loads(marker.read_text())['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Embedding comparison already has a live worker.')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(config_path)]
            with (output/'run.log').open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(marker,{'pid':process.pid,'command':command,'log':str((output/'run.log').relative_to(ROOT))})
            print(f'Worker PID {process.pid}. Progress: {output/"run.log"}',flush=True)
        return 0
    with locked(output/'worker.lock',blocking=False):
        comparison=None
        try:
            atomic_json(output/'status.json',{'phase':'initializing','status':'running','pid':os.getpid(),
                'updated_at':datetime.now(timezone.utc).isoformat()})
            print('[comparison] Verifying frozen inputs and preparing the validation-only extension',flush=True)
            comparison=Comparison(config);comparison.run()
        except (Exception,KeyboardInterrupt) as exc:
            error=redact(str(exc));state={'phase':'initializing','status':'incomplete','error_type':type(exc).__name__,'error':error[:1000],
                'updated_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
            current=output/'status.json'
            if current.exists():state={**json.loads(current.read_text()),**{k:v for k,v in state.items() if k!='phase'}}
            atomic_json(current,state)
            print(f'[comparison stopped] {type(exc).__name__}: {error[:1000]}',flush=True)
            print('Successful embedding and model batches are saved. Relaunch this command to resume.',flush=True)
            if comparison is not None:comparison.report()
            return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
