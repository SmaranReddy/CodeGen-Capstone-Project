"""Experimental wrapper for MQL-0008; does not alter frozen benchmark files."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'kaggle/nl2mql_mql0008_approved_corpus_v2_1/src'
sys.path.insert(0, str(SRC))
import mql0008_core as core
from mql_guard import augment_user_prompt, check_mql

VERSION = 'syntax-guard-v1'


def augment_frozen(prompts):
    """Copy, augment both arms' common base, and recompute prompt identity."""
    result = copy.deepcopy(prompts)
    for p in result:
        p['messages'][0]['content'] = augment_user_prompt(p['messages'][0]['content'])
        p['prompt_version'] = p['prompt_version'] + '+' + VERSION
        p['template_sha256'] = core.sha256_text(p['template_sha256'] + '+' + VERSION)
        p['prompt_sha256'] = core.prompt_sha(p['system'], p['messages'])
    return result


def build_experimental_prompts(cfg, arm, eval_rows, frozen_prompts, docs, eval_infos, count_tokens):
    base = augment_frozen(frozen_prompts)
    ctx = cfg['context']
    for p in base:
        total = count_tokens(p['system']) + count_tokens(p['messages'][0]['content'])
        core.need(total + ctx['max_new_tokens'] + ctx['margin_tokens'] <= ctx['n_ctx'], 'Augmented base exceeds context budget')
    plan = core.build_plan(cfg, arm, eval_rows, base, docs, eval_infos, count_tokens)
    prompts = core.prompts_for_arm(plan, base, {d['doc_id']: d for d in docs})
    return plan, prompts


def diagnostic_generator(generate_batch):
    """Log diagnostics without changing predictions or executing any queries."""
    def wrapped(batch):
        results = []
        for text, meta, error in generate_batch(batch):
            meta = dict(meta or {})
            if error is None:
                try:
                    issues = check_mql(json.loads(text))
                except (ValueError, TypeError):
                    issues = [{'code': 'JSON_PARSE', 'path': 'output'}]
                meta['syntax_guard'] = {'version': VERSION, 'issues': issues}
            results.append((text, meta, error))
        return results
    return wrapped


def smoke_test():
    text = 'Schema: toy(age)\n\nQuestion: Find adults.'
    fp = {'id':'toy1','db_id':'toy','system':'Generate JSON.', 'messages':[{'role':'user','content':text}], 'prompt_version':'toy-v1','template_sha256':'toy-template'}
    fp['prompt_sha256'] = core.prompt_sha(fp['system'], fp['messages'])
    cfg = {'arms': {core.ARMS[0]: {'retrieval':'none','k':0}, core.ARMS[1]:{'retrieval':'hybrid_masked','k':3}}, 'context':{'max_context_tokens':1500,'n_ctx':8192,'max_new_tokens':1024,'margin_tokens':64}, 'retrieval':{'hybrid_weight':1.0}}
    doc = {'doc_id':'example1','db_id':'other','kind':'example','question':'Find adults','masked_question':'find adults','mql':{'collection':'other','pipeline':[]},'patterns':[],'content':'Question: Find adults\nMQL: {}'}
    rows = [{'id':'toy1','db_id':'toy','question':'Find adults.'}]
    # Mock retrieval only: exercise real budget, planning, rebuilding and hashing.
    from unittest.mock import patch
    class ToyRetriever:
        def rank_examples(self,*args,**kwargs): return [(doc,1.0)]
    count = lambda s: len(s.split())  # toy counter, not actual tokenizer
    with patch.object(core,'make_retriever',side_effect=lambda cfg,arm,docs: None if arm==core.ARMS[0] else ToyRetriever()):
        pa, a = build_experimental_prompts(cfg,core.ARMS[0],rows,[fp],[doc],{'toy':{}},count)
        pb, b = build_experimental_prompts(cfg,core.ARMS[1],rows,[fp],[doc],{'toy':{}},count)
    from mql_guard import RULES
    assert all(RULES in p[0]['messages'][0]['content'] for p in (a,b))
    assert fp['messages'][0]['content']==text
    assert a[0]['prompt_sha256'] != fp['prompt_sha256']
    assert a[0]['prompt_sha256']==pa[0]['prompt_sha256'] and b[0]['prompt_sha256']==pb[0]['prompt_sha256']
    block = core.RB.render_block('toy',[],[doc],[])
    assert b[0]['messages'][0]['content'] == core.RB.insert_block(a[0]['messages'][0]['content'],block)
    print('Both arms, frozen input preservation, prompt hashes and parity: PASS')
    bad = json.dumps({'collection':'toy','pipeline':[{'match':{}}]})
    good = json.dumps({'collection':'toy','pipeline':[{'$match':{'age':20}}]})
    original = [(bad,{},None),(good,{},None)]
    result = diagnostic_generator(lambda batch: original)([{},{}])
    assert result[0][1]['syntax_guard']['issues'][0]['code']=='C3'
    assert result[1][1]['syntax_guard']['issues']==[]
    assert [r[0] for r in result]==[bad,good] and original[0][1]=={}
    print('Mock generation diagnostics and unchanged predictions: PASS')
    print('Integration smoke test passed (no model or database calls).')


if __name__ == '__main__':
    smoke_test()
