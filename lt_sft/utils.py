import torch
from transformers import AutoModelForSeq2SeqLM

def get_mask(pretrained_model = "Helsinki-NLP/opus-mt-es-fi", finetuned_model = "americasnlp-lct-ehu/es_fi_quz", K = 10000):
    # 1a. Load the pre-trained Spanish-Finnish model
    model_es_fi = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

    # 1b. Load the pre-trained Spanish-Finnish model
    model_es_quz = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model)

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