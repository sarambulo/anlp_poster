import torch
from transformers import AutoModelForSeq2SeqLM
from typing import Literal, Dict, Sequence

def get_mask_layerwise(
    pretrained_model: str = "Helsinki-NLP/opus-mt-es-fi",
    finetuned_model: str = "americasnlp-lct-ehu/es_fi_quz",
    layer_pct: Dict[str, Sequence[float]] = {
        "encoder": [1, 1, 1, 1, 1, 1],
        "decoder": [1, 1, 1, 1, 1, 1]
    },
) -> Dict[str, torch.Tensor]:
    """
    Build a binary mask dict over parameters, selecting per-layer top-changes
    according to `layer_pct`.

    layer_pct must be of the form:
      {
        "encoder": [p0, p1, ..., p{L_enc-1}],
        "decoder": [q0, q1, ..., q{L_dec-1}]
      }
    where each p_i and q_j is the fraction of THAT
    layer's parameters to keep (1.0 ⇒ keep all, 0 ⇒ freeze all).

    Note: in our case, L_enc = L_dec = 6

    By default:
     - lm_head.*                → always kept (mask=1)
     - any "...layer_norm..."   → always frozen (mask=0)
     - all other keys           → masked per-layer (or frozen if not in any layer)
    """
    # 1) load models & state dicts
    model_es_fi = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
    model_es_quz  = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model)
    state_es_fi = model_es_fi.state_dict()
    state_es_quz = model_es_quz.state_dict()
    if set(state_es_fi.keys()) != set(state_es_quz.keys()):
        raise ValueError("Checkpoint key mismatch")

    # 2) helper to gather all diffs for a given prefix list of keys
    def _collect_diffs(prefix_match):
        keys = [k for k in state_es_fi if any(k.startswith(pref) for pref in prefix_match)]
        diffs = {k: (state_es_quz[k] - state_es_fi[k]).abs().view(-1) for k in keys}
        flat = torch.cat(list(diffs.values()))
        return diffs, flat, keys

    # 3) build an empty mask space
    full_mask: Dict[str, torch.Tensor] = {}
    for name, tensor in state_es_fi.items():
        full_mask[name] = torch.zeros_like(tensor)

    # 4) always keep head, always freeze any layer_norm
    for name in state_es_fi:
        if name in ("lm_head.weight", "final_logits_bias"):
            full_mask[name] = torch.ones_like(state_es_fi[name])
        elif "layer_norm" in name:
            full_mask[name] = torch.zeros_like(state_es_fi[name])

    # 5) encoder layers
    enc_pcts = layer_pct.get("encoder", [])
    for i, pct in enumerate(enc_pcts):
        prefix = [f"model.encoder.layers.{i}."]
        diffs, flat, keys = _collect_diffs(prefix)
        K = int(flat.numel() * pct)
        if K <= 0:
            continue
        topk_idx = torch.topk(flat, k=K, sorted=False).indices
        mask_flat = torch.zeros_like(flat)
        mask_flat[topk_idx] = 1.0

        # split back
        ptr = 0
        for k in keys:
            n = diffs[k].numel()
            m = mask_flat[ptr:ptr+n].view(state_es_fi[k].shape)
            full_mask[k] = m.to(state_es_fi[k].device)
            ptr += n

    # 6) decoder layers
    dec_pcts = layer_pct.get("decoder", [])
    for j, pct in enumerate(dec_pcts):
        prefix = [f"model.decoder.layers.{j}."]
        diffs, flat, keys = _collect_diffs(prefix)
        K = int(flat.numel() * pct)
        if K <= 0:
            continue
        topk_idx = torch.topk(flat, k=K, sorted=False).indices
        mask_flat = torch.zeros_like(flat)
        mask_flat[topk_idx] = 1.0

        ptr = 0
        for k in keys:
            n = diffs[k].numel()
            m = mask_flat[ptr:ptr+n].view(state_es_fi[k].shape)
            full_mask[k] = m.to(state_es_fi[k].device)
            ptr += n

    return full_mask

def get_mask_granular(
    pretrained_model: str = "Helsinki-NLP/opus-mt-es-fi",
    finetuned_model: str = "americasnlp-lct-ehu/es_fi_quz",
    K_pct: float = 1.0,
    part: Literal["all", "encoder", "decoder"] = "all",
) -> Dict[str, torch.Tensor]:
    """
    Compute a binary mask over parameters, selecting the top K_pct parameters
    by absolute change between the pretrained and finetuned checkpoints.

    If part=="encoder", only consider parameters whose name starts with
    "model.encoder."; if part=="decoder", only those under "model.decoder."
    (plus the final lm_head); if "all", consider everything.

    Returns a dict mapping every parameter name to a mask tensor of the same
    shape (1.0 = update, 0.0 = freeze).
    """

    # 1. Load both checkpoints
    model_es_fi   = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
    model_es_quz  = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model)

    # 2. Extract state_dicts
    state_es_fi   = model_es_fi.state_dict()
    state_es_quz  = model_es_quz.state_dict()

    # 3. Pick keys to finetune
    if part == "encoder":
        keys_to_finetune = [k for k in state_es_fi if k.startswith("model.encoder.")]
    elif part == "decoder":
        keys_to_finetune = [
            k for k in state_es_fi
            if k.startswith("model.decoder.") or k in ("lm_head.weight", "final_logits_bias")
        ]
    elif part == "all":
        keys_to_finetune = list(state_es_fi.keys())
    else:
        raise ValueError(f"Invalid part={part!r}, must be 'all','encoder','decoder'")

    # 4. Sanity checks
    if set(state_es_fi.keys()) != set(state_es_quz.keys()):
       raise ValueError("Checkpoint key mismatch")

    for k in keys_to_finetune:
        if state_es_fi[k].shape != state_es_quz[k].shape:
            raise ValueError(f"Shape mismatch for {k}")

    # 5. Compute flattened diffs only for chosen keys
    diffs = {k: (state_es_quz[k] - state_es_fi[k]).abs().view(-1) for k in keys_to_finetune}

    # 6. Select top-K_pct of subset of keys being finetuned
    all_diffs    = torch.cat(list(diffs.values()))
    total_params = all_diffs.numel()
    K            = int(total_params * K_pct)
    topk_vals, topk_idx = torch.topk(all_diffs, k=K, sorted=False)

    # 7. Build flat mask
    flat_mask = torch.zeros_like(all_diffs)
    flat_mask[topk_idx] = 1.0

    # 8. Split back into per-param masks
    mask_dict_ranked: Dict[str, torch.Tensor] = {}
    pointer = 0
    for k, chunk in diffs.items():
        numel = chunk.numel()
        mask_dict_ranked[k] = flat_mask[pointer : pointer + numel].view(state_es_fi[k].shape)
        pointer += numel

    # 9. Build full mask, then override head & layer‑norms
    full_mask: Dict[str, torch.Tensor] = {}
    for layer_name, tensor in state_es_fi.items():
        if layer_name in ("lm_head.weight", "final_logits_bias"):
            # always fully fine-tune the head (in LT-SFT paper, they fully fine-tune the classifier head)
            full_mask[layer_name] = torch.ones_like(tensor)
        elif "layer_norm" in layer_name:
            # freeze every layer-norm (in LT-SFT paper, they fix the layer norm parameters)
            full_mask[layer_name] = torch.zeros_like(tensor)
        elif layer_name in mask_dict_ranked:
            full_mask[layer_name] = mask_dict_ranked[layer_name].to(tensor.device)
        else:
            full_mask[layer_name] = torch.zeros_like(tensor)
    
    # TODO: in the LT-SFT paper they also fix the params of the output embedding matrix. Maybe we should do this (analogously)

    return full_mask


def get_mask(pretrained_model = "Helsinki-NLP/opus-mt-es-fi", finetuned_model = "americasnlp-lct-ehu/es_fi_quz", K_pct=1):
    # 1a. Load the pre-trained Spanish-Finnish model
    model_es_fi = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

    # 1b. Load the pre-trained Spanish-Finnish model
    model_es_quz = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model)

    total_params = sum(p.numel() for p in model_es_fi.parameters() if p.requires_grad)
    K = int(total_params * K_pct)

    # 2. Extract their state dictionaries
    state_fi = model_es_fi.state_dict()
    state_quz = model_es_quz.state_dict()

    # Sanity check
    if state_fi.keys() != state_quz.keys():
        missing_in_fi = state_quz.keys() - state_fi.keys()
        missing_in_quz = state_fi.keys() - state_quz.keys()
        raise ValueError(
            f"State dict key mismatch:\n"
            f"  Missing in Spanish–Finnish: {missing_in_fi}\n"
            f"  Missing in Spanish–Quechua: {missing_in_quz}"
        )

    # Sanity check
    for name in state_fi:
        if state_fi[name].shape != state_quz[name].shape:
            raise ValueError(
                f"Shape mismatch for parameter '{name}':\n"
                f"  Spanish–Finnish shape: {state_fi[name].shape}\n"
                f"  Spanish–Quechua shape: {state_quz[name].shape}"
            )

    # 3. Compute the absolute differences for each parameter
    diffs = {name: (state_quz[name] - state_fi[name]).abs() for name in state_fi if name in state_quz}

    # 4. Flatten all differences into a single vector
    all_diffs = torch.cat([tensor.view(-1) for tensor in diffs.values()])

    # 5. Identify the top K parameters (by absolute change)
    topk_vals, topk_idxs = torch.topk(all_diffs, K)

    # 6. Create a binary mask μ for each parameter tensor
    mask = torch.zeros_like(all_diffs)
    mask[topk_idxs] = 1.0

    # 7. Split the mask back into the original parameter shapes
    mask_dict = {}
    pointer = 0
    for name, tensor in diffs.items():
        numel = tensor.numel()
        mask_dict[name] = mask[pointer : pointer + numel].view(tensor.shape)
        pointer += numel

    # mask_dict now contains, for each parameter name, a binary tensor μ[name]
    # indicating which of the K parameters changed most significantly.
    return mask_dict