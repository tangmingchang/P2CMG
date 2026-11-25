"""Validate evidence references and render identical musical facts two ways.

This implements a data interface, not a trained novel generator. It deliberately
does not equate model self-confidence with verified visual-musical correspondence.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ORDER=('mood','tempo','instrumentation','texture','spatiality','dynamics','mode')

def validate(record):
    required={'case_id','provenance','visual_evidence','interpretations','music_proposals','unknowns'}
    if set(record)!=required:raise ValueError('Missing or extra top-level semantic fields')
    if not isinstance(record['case_id'],str) or not record['case_id']:raise ValueError('Case ID required')
    origin=record['provenance']
    if origin.get('author_type') not in ('AI_draft','human','adjudicated_human'):raise ValueError('Unknown author type')
    if origin.get('human_review_status') not in ('pending','reviewed','disputed'):raise ValueError('Unknown review status')
    if origin.get('image_access') not in ('full_image','contact_sheet_only'):raise ValueError('Unknown image access')
    evidence={}
    for item in record['visual_evidence']:
        if item['id'] in evidence:raise ValueError('Repeated evidence ID')
        if not item.get('observation') or item.get('certainty') not in ('clear','ambiguous'):raise ValueError('Invalid observation')
        box=item.get('region_xyxy_normalized')
        if box is not None and (len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in box) or not (box[0]<box[2] and box[1]<box[3])):raise ValueError('Invalid region box')
        evidence[item['id']]=item
    interpretations={}
    for item in record['interpretations']:
        if item['id'] in interpretations:raise ValueError('Repeated interpretation ID')
        if not item.get('evidence_ids') or not set(item['evidence_ids'])<=set(evidence):raise ValueError('Interpretation references missing evidence')
        if not item.get('interpretation') or not item.get('alternative'):raise ValueError('Interpretation and alternative required')
        interpretations[item['id']]=item
    attributes=set()
    for item in record['music_proposals']:
        if item['attribute'] not in ORDER or item['attribute'] in attributes:raise ValueError('Unknown or duplicated control attribute')
        attributes.add(item['attribute'])
        if not set(item.get('evidence_ids',[]))<=set(evidence) or not set(item.get('interpretation_ids',[]))<=set(interpretations):raise ValueError('Control has unresolved evidence or interpretation references')
        if item['status']=='unknown':
            if item['value'] is not None:raise ValueError('Unknown control must retain null value')
        elif item['status']=='interpretive_choice':
            if not item.get('evidence_ids') or not item.get('interpretation_ids') or not item.get('rationale'):raise ValueError('Interpretive control needs evidence, interpretation and rationale')
            if not isinstance(item['value'],str) or not item['value'].strip():raise ValueError('Missing control value')
            linked=set().union(*(set(interpretations[i]['evidence_ids']) for i in item['interpretation_ids']))
            if not set(item['evidence_ids'])&linked:raise ValueError('Control evidence must overlap its interpretation evidence')
        elif item['status']=='user_specified':
            if not isinstance(item['value'],str) or not item['value'].strip():raise ValueError('User-specified control needs a value')
        else:raise ValueError('Unknown control status')
    if not all(isinstance(x,str) for x in record['unknowns']):raise ValueError('Unknowns must be strings')
    return record

def render(record):
    validate(record)
    active={p['attribute']:p for p in record['music_proposals'] if p['status']!='unknown'}
    # Exactly the same attribute labels and values; only punctuation/line layout changes.
    facts=[f"{key.capitalize()}: {active[key]['value'].strip().rstrip('.')}" for key in ORDER if key in active]
    return {'case_id':record['case_id'],
        'plain':'Create instrumental music. '+'. '.join(facts)+'.',
        'structured':'Create instrumental music.\n'+'\n'.join(facts)+'.',
        'facts':{key:active[key]['value'] for key in ORDER if key in active},
        'uncertainty_notice':'Unknown attributes omitted, not imputed. Interpretive choices are proposals, not measured properties.',
        'provenance':record['provenance']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=json.loads(a.input.read_text(encoding='utf-8'));result=render(value)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Validated references and wrote two renderings with identical attribute values. No model was run.')
