import torch

from memseg_agent.working_memory import WorkingMemoryBuffer


class DummyEmbedder:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image.flatten()[:2].float()


class DummyModel:
    target_class = 1

    def encode_observation(self, image, mask):
        return torch.ones(2, 1, 2), torch.zeros(2, 1, 2)


def test_memory_add_retrieve_and_delta_update() -> None:
    memory = WorkingMemoryBuffer(DummyEmbedder(), min_foreground_pixels=1, add_similarity_threshold=0.5)
    image = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])
    update = memory.update(DummyModel(), image, torch.ones(1, 1), dice=0.0)
    assert update["action"] == "added"
    assert len(memory.retrieve(image)) == 1
    update = memory.update(DummyModel(), image, torch.ones(1, 1), dice=0.2)
    assert update["action"] == "ema_updated"
