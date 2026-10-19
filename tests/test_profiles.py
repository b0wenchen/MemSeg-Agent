import torch
from memseg_agent.model import MemSegAgent


def test_cross_domain_static_mask_is_never_scaled():
    # Exercise real static_tokens without downloading or constructing SAM2.
    model = MemSegAgent.__new__(MemSegAgent)
    torch.nn.Module.__init__(model)
    model.sam2 = torch.nn.Identity()
    model.memory_mode = "pseudo_observation"
    model.eval_profile = "cross_domain"
    model.mem_img_logits = torch.nn.Parameter(torch.zeros(1))
    model.mem_mask_logits = torch.nn.Parameter(torch.ones(1))
    model._tokens_from_memory_encoder = lambda image, mask: (image, mask)
    model.train()
    assert torch.equal(model.static_tokens()[1], torch.sigmoid(model.mem_mask_logits))
    model.eval()
    assert torch.equal(model.static_tokens()[1], torch.sigmoid(model.mem_mask_logits))
    model.eval_profile = "training"
    assert torch.equal(model.static_tokens()[1], torch.sigmoid(model.mem_mask_logits))
    # Corrected observations follow a different encoder input convention:
    # foreground -> +10, background -> -10, in both evaluation profiles.
    image = torch.zeros(3, 2, 2)
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    for profile in ("training", "cross_domain"):
        model.eval_profile = profile
        _, encoded_mask = model.encode_observation(image, mask)
        assert encoded_mask[0, 0, 0, 0].item() == -10
        assert encoded_mask[0, 0, 0, -1].item() == 10
