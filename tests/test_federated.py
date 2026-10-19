import torch

from memseg_agent.federated import fedavg_static_memories


def test_fedavg_static_memory_only() -> None:
    one = {"mem_img_logits": torch.tensor([1.0]), "mem_mask_logits": torch.tensor([3.0])}
    two = {"mem_img_logits": torch.tensor([5.0]), "mem_mask_logits": torch.tensor([7.0])}
    result = fedavg_static_memories([one, two], [1, 3])
    assert torch.allclose(result["mem_img_logits"], torch.tensor([4.0]))
    assert torch.allclose(result["mem_mask_logits"], torch.tensor([6.0]))
