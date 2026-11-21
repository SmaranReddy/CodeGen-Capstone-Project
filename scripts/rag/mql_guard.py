"""Experimental prompt augmentation and limited static checks for NL-to-MQL.
No database, model calls, or gold answers are needed. Not a full validator.
"""
import json

RULES = '''MQL syntax checklist:
- In $match use field predicates, e.g. {"age":{"$gt":20}}. Expression comparisons use $expr, e.g. {"$expr":{"$gt":["$age",20]}}.
- In $group every output except _id must be an accumulator, e.g. {"name":{"$first":"$name"}}; do not use bare field references or arithmetic as accumulators.
- Every pipeline stage name and every $unwind field path must start with $.
- Query $in lists contain values, not aggregation stages. Use separate pipeline stages for joins.
Return only the requested JSON.'''


def augment_user_prompt(text):
    """Apply identically to zero-shot and RAG messages, before the final question."""
    if RULES in text:
        return text
    head, marker, tail = text.rpartition('\n\nQuestion: ')
    if not marker:
        raise ValueError('Expected final Question: marker')
    return head + '\n\n' + RULES + marker + tail


COMPARISONS = {'$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$regexMatch', '$cmp'}
ACCUMULATORS = {'$sum', '$avg', '$min', '$max', '$first', '$last', '$push', '$addToSet', '$count', '$stdDevPop', '$stdDevSamp', '$mergeObjects', '$firstN', '$lastN', '$minN', '$maxN', '$top', '$bottom', '$topN', '$bottomN', '$accumulator', '$median', '$percentile', '$concatArrays', '$setUnion'}
STAGES = {'$match', '$group', '$project', '$lookup', '$unwind', '$sort', '$limit', '$skip', '$count', '$addFields', '$set', '$unset', '$replaceRoot', '$replaceWith', '$facet', '$unionWith', '$bucket', '$bucketAuto', '$sortByCount', '$sample', '$out', '$merge', '$setWindowFields', '$graphLookup', '$redact'}


def check_mql(query):
    """Return diagnostic codes and locations for a limited set of syntax errors."""
    issues = []
    def flag(code, path):
        issues.append({'code': code, 'path': path})
    def match(node, path):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in COMPARISONS:
                flag('C1', path + '.' + key)
            elif key in ('$and', '$or', '$nor') and isinstance(value, list):
                for i, item in enumerate(value):
                    match(item, f'{path}.{key}[{i}]')
            elif not key.startswith('$') and isinstance(value, dict):
                values = value.get('$in')
                if isinstance(values, list) and any(isinstance(v, dict) and STAGES.intersection(v) for v in values):
                    flag('C9', path + '.' + key + '.$in')
    def pipeline(stages, path):
        if not isinstance(stages, list):
            flag('SHAPE', path)
            return
        for i, stage in enumerate(stages):
            p = f'{path}[{i}]'
            if not isinstance(stage, dict) or len(stage) != 1:
                flag('C4', p)
                continue
            op, body = next(iter(stage.items()))
            if not op.startswith('$'):
                flag('C3' if '$' + op in STAGES else 'C4', p)
                continue
            if op == '$match':
                match(body, p + '.$match')
            elif op == '$group' and isinstance(body, dict):
                for key, val in body.items():
                    if key == '_id':
                        continue
                    loc = p + '.$group.' + key
                    if not isinstance(val, dict) or len(val) != 1:
                        flag('C2', loc)
                    elif next(iter(val)) not in ACCUMULATORS:
                        flag('C8', loc)
            elif op == '$unwind':
                field = body.get('path') if isinstance(body, dict) else body
                if not isinstance(field, str) or not field.startswith('$'):
                    flag('C5', p + '.$unwind')
            elif op in ('$lookup', '$unionWith') and isinstance(body, dict) and 'pipeline' in body:
                pipeline(body['pipeline'], p + '.' + op + '.pipeline')
            elif op == '$facet' and isinstance(body, dict):
                for name, sub in body.items():
                    pipeline(sub, p + '.$facet.' + name)
    if not isinstance(query, dict) or not isinstance(query.get('collection'), str):
        flag('SHAPE', 'collection')
    else:
        pipeline(query.get('pipeline'), 'pipeline')
    return issues


def self_test():
    # Synthetic toy-schema fixtures, not benchmark predictions.
    fixtures = {
        'C1': [{'$match': {'$and': [{'$gt': ['$age', 20]}]}}],
        'C2': [{'$group': {'_id': '$team', 'name': '$name'}}],
        'C3': [{'match': {'age': 20}}],
        'C4': [{'name': 'Alice', 'age': 20}],
        'C5': [{'$unwind': 'members'}],
        'C8': [{'$group': {'_id': None, 'x': {'$subtract': ['$a', '$b']}}}],
        'C9': [{'$match': {'id': {'$in': [{'$lookup': {'from': 'other'}}]}}}],
    }
    valid = [
        [{'$match': {'age': {'$gt': 20}}}],
        [{'$match': {'$expr': {'$gt': ['$age', 20]}}}],
        [{'$group': {'_id': '$team', 'name': {'$first': '$name'}, 'n': {'$sum': 1}}}],
        [{'$unwind': '$members'}],
        [{'$match': {'id': {'$in': [1, 2]}}}],
        [{'$lookup': {'from': 'other', 'pipeline': [{'$match': {'$expr': {'$eq': ['$a', '$b']}}}], 'as': 'joined'}}],
    ]
    passed = 0
    for expected, stages in fixtures.items():
        codes = {x['code'] for x in check_mql({'collection': 'toy', 'pipeline': stages})}
        ok = expected in codes
        print(f'{expected}: {"PASS" if ok else "FAIL"} {sorted(codes)}')
        assert ok
        passed += 1
    for i, stages in enumerate(valid, 1):
        issues = check_mql({'collection': 'toy', 'pipeline': stages})
        print(f'valid {i}: {"PASS" if not issues else "FAIL"} {issues}')
        assert not issues
        passed += 1
    base = 'Schema: toy(age)\n\nQuestion: Find adults.'
    rag = 'Schema: toy(age)\n\nReference: example\n\nQuestion: Find adults.'
    a, b = augment_user_prompt(base), augment_user_prompt(rag)
    assert b.replace('\n\nReference: example', '') == a
    assert augment_user_prompt(a) == a
    print('Prompt parity and idempotence: PASS')
    print(f'{passed}/{len(fixtures)+len(valid)} syntax fixtures passed; 2 prompt assertions passed.')


if __name__ == '__main__':
    self_test()
