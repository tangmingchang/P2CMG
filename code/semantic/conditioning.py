"""Real independent-field conditioning, with equal capacity and fixed token slots.

Both paragraph and field routes expose 448 text slots + one visual slot to the
same LM and projections. Frozen T5 encodes either the whole paragraph once or
each of the seven labelled attributes independently. No invented field values.
"""
from __future__ import annotations
import torch
from torch import nn

FIELDS = ('mood', 'tempo', 'instrumentation', 'texture', 'spatiality', 'dynamics', 'mode')
FIELD_TOKENS = 64
TEXT_TOKENS = len(FIELDS) * FIELD_TOKENS


def text_views(facts):
    if set(facts) != set(FIELDS):
        raise ValueError('All seven fields required; unknown fields must be null')
    if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in facts.values()):
        raise ValueError('Fields are nonempty strings or explicit null')
    labelled = [f'{f.capitalize()}: {facts[f].strip().rstrip(".")}' if facts[f] is not None else '' for f in FIELDS]
    return '. '.join(v for v in labelled if v) + '.', labelled


class SlotConditioner(nn.Module):
    """Feature tensors are frozen cache entries; only the shared projections learn."""
    def __init__(self, text_projection, image_dim=512):
        super().__init__()
        self.text_projection = text_projection
        self.image_projection = nn.Linear(image_dim, text_projection.out_features)

    def forward(self, text, text_mask, image, image_mask):
        if text.ndim != 3 or text.shape[1] != TEXT_TOKENS:
            raise ValueError('All routes must have 448 fixed text slots')
        if text_mask.shape != text.shape[:2] or image.shape[0] != text.shape[0]:
            raise ValueError('Condition batch/mask shape mismatch')
        image_mask = image_mask.reshape(-1, 1)
        # Mask AFTER the affine projection: bias must not leak a missing modality.
        t = self.text_projection(text) * text_mask[..., None]
        v = self.image_projection(image).unsqueeze(1) * image_mask[..., None]
        return torch.cat((t, v), dim=1), torch.cat((text_mask, image_mask), dim=1)


def assemble(features, records, encoding, use_image, device):
    if encoding not in ('paragraph', 'fields'):
        raise ValueError('Unknown encoding route')
    text = torch.stack([features[r['semantic_key']][encoding]['embedding'] for r in records]).to(device).float()
    mask = torch.stack([features[r['semantic_key']][encoding]['mask'] for r in records]).to(device).float()
    images = torch.stack([features['images'][r['image_key']] for r in records]).to(device).float()
    imask = torch.tensor([float(use_image and not r.get('drop_condition', False)) for r in records], device=device)
    for i, r in enumerate(records):
        if r.get('drop_condition', False):
            mask[i].zero_()
    return text, mask, images, imask


@torch.no_grad()
def encode_views(t5, tokenizer, facts, device):
    paragraph, fields = text_views(facts)
    result = {}
    for name, texts, cap in [('paragraph', [paragraph], TEXT_TOKENS), ('fields', fields, FIELD_TOKENS)]:
        active = [bool(t) and t != '.' for t in texts]
        # Never silently truncate a field or a paragraph: information matching
        # would no longer hold. A failure is saved and must be resolved upstream.
        ids = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        lengths = ids.attention_mask.sum(1).tolist()
        if any(n > cap for n, enabled in zip(lengths, active) if enabled):
            raise ValueError(f'{name} exceeds token budget; do not silently truncate: {lengths}')
        inp = tokenizer(texts, padding='max_length', truncation=True, max_length=cap, return_tensors='pt').to(device)
        emb = t5(**inp).last_hidden_state
        mask = inp.attention_mask.float()
        for i, enabled in enumerate(active):
            if not enabled:
                mask[i].zero_()
        result[name] = {'embedding': (emb * mask[..., None]).reshape(TEXT_TOKENS, -1).cpu().half(),
                        'mask': mask.reshape(TEXT_TOKENS).cpu().bool(),
                        'token_ids': inp.input_ids.cpu().tolist(),
                        'active_token_lengths': [n if enabled else 0 for n, enabled in zip(lengths, active)]}
    return result
