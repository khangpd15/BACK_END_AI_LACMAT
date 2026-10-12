"""Frozen inference subset imported from the training workspace."""
def expected_center(logits, valid_mask, torch):
    h, w = logits.shape[-2:]
    probs = logits.masked_fill(~valid_mask, -1000000000.0).flatten(1).softmax(-1).reshape_as(logits)
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device, dtype=logits.dtype), torch.arange(w, device=logits.device, dtype=logits.dtype), indexing='ij')
    return torch.stack(((probs * xx).sum((1, 2)), (probs * yy).sum((1, 2))), 1)
