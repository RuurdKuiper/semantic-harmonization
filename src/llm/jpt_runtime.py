"""JPT-4B through its published llm2jev scoring interface on a local GPU."""
from functools import lru_cache
import platform


class AppleJPT:
    """Published llm2jev prompts and probabilities with MLX model execution."""
    def __init__(self, model_name):
        from mlx_vlm import load
        from src.llm.model_paths import prepared_model_path
        print(f'[JPT] Loading {model_name} with MLX (original BF16 weights)',flush=True)
        self.model,self.processor=load(prepared_model_path(model_name))
        self.prefix_tokens=[];self.prefix_cache=None

    def warm(self,prefix,images):
        if images:raise ValueError('The study uses text-only questions.')
        import mlx.core as mx
        # Leave a boundary margin so tokenization cannot merge the final prefix
        # token with the question. Every scoring request verifies this prefix.
        tokens=self.processor.tokenizer.encode(prefix,add_special_tokens=False)[:-2]
        if tokens==self.prefix_tokens:return
        self.prefix_tokens=tokens;lm=self.model.language_model
        self.prefix_cache=lm.make_cache()
        if tokens:
            lm(mx.array([tokens]),cache=self.prefix_cache,skip_logits=True)
            mx.eval([c.state for c in self.prefix_cache])

    def score(self,text,images,ids):
        if images:raise ValueError('The study uses text-only questions.')
        rows,count=self._score_texts([text],ids)
        return rows[0],count

    def _score_texts(self,texts,ids):
        import mlx.core as mx
        tokens=[self.processor.tokenizer.encode(text,add_special_tokens=False) for text in texts]
        lm=self.model.language_model
        cache=lm.make_cache();suffixes=tokens
        if self.prefix_tokens and all(t[:len(self.prefix_tokens)]==self.prefix_tokens for t in tokens):
            from mlx.utils import tree_map
            for fresh,saved in zip(cache,self.prefix_cache):
                snapshot=tree_map(lambda x:mx.repeat(x,len(texts),axis=0) if isinstance(x,mx.array) else x,
                                  saved.prefix_cache_snapshot())
                fresh.prefix_cache_restore(snapshot)
            suffixes=[t[len(self.prefix_tokens):] for t in tokens]
        lengths=[len(s) for s in suffixes];width=max(lengths)
        # Future padding cannot affect earlier positions in this causal model.
        # Select each row's final real token before projecting to vocabulary.
        padded=[s+[0]*(width-len(s)) for s in suffixes]
        out=lm(mx.array(padded),cache=cache,skip_logits=True,return_hidden=True)
        hidden=out.hidden_states[-1][mx.arange(len(texts)),mx.array(lengths)-1][:,None,:]
        logits=lm.speculative_logits_from_hidden(hidden)[:,0].astype(mx.float32)
        values=(logits-mx.logsumexp(logits,axis=-1,keepdims=True))[:,mx.array(ids)].tolist()
        return values,sum(map(len,tokens))

    def score_many(self,texts,ids):
        rows=[];count=0
        for i in range(0,len(texts),4):
            batch,n=self._score_texts(texts[i:i+4],ids);rows.extend(batch);count+=n
            print(f'[JPT] {min(i+4,len(texts))}/{len(texts)} questions scored',flush=True)
        return rows,count


@lru_cache(maxsize=1)
def load_jpt(model_name):
    from llm2jev.engine import LLM2Jev
    if platform.system()=='Darwin' and platform.machine()=='arm64':
        backend=AppleJPT(model_name)
        return LLM2Jev(backend.processor,backend,temperature=1.036,workers=1)
    import torch
    from llm2jev.backends import HF

    class LastTokenHF(HF):
        def __init__(self, model, device=None, **kwargs):
            # Loading safetensors directly onto MPS starts concurrent transfer
            # threads. Load on CPU and transfer serially to avoid a native
            # allocator/Metal crash in that loading path on this Mac.
            super().__init__(model, device='cpu' if device=='mps' else device, **kwargs)
            if device=='mps':self.model.to(device)
        # Restrict the LM projection to the last position. The upstream HF
        # reference projects every EDF token, wasting substantial GPU memory.
        def score(self, text, images, ids):
            if images:
                raise ValueError('This experiment uses text-only JPT inputs.')
            rows, count = self.score_many([text], ids)
            return rows[0], count

        def score_many(self, texts, ids):
            self.tok.padding_side = 'left'
            rows = []; count = 0
            for start in range(0, len(texts), 4):
                inputs = self.tok(texts[start:start+4],padding=True,return_tensors='pt',
                                  add_special_tokens=False)
                count += int(inputs['attention_mask'].sum())
                inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
                with self.lock, torch.inference_mode():
                    logits = self.model(**inputs,logits_to_keep=1).logits[:,-1].float().log_softmax(-1)
                rows.extend(logits[:,ids].cpu().tolist())
            return rows,count

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[JPT] Loading {model_name} on {device}',flush=True)
    from src.llm.model_paths import prepared_model_path
    backend = LastTokenHF(prepared_model_path(model_name),device=device,dtype='bfloat16')
    return LLM2Jev(backend.processor or backend.tok,backend,temperature=1.036,workers=1)


def predict_jpt(state, questions, model_name):
    answers, usage = load_jpt(model_name).run(state, questions)
    return {'answers':answers,'model':model_name,'usage':usage,
            'runtime':'llm2jev/mlx' if platform.system()=='Darwin' and platform.machine()=='arm64' else 'llm2jev/hf',
            'temperature':1.036}
