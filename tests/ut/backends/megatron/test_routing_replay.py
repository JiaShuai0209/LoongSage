"""Unit tests for the routing-replay helpers in coda/backends/megatron/model.py.

Uses the real ``RouterReplay`` (Megatron-Core must be installed) on a fake model chunk whose
modules carry ``router_replay`` attributes; layer layout and data packing are patched.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction  # noqa: E402

from coda.backends.megatron import model as model_mod  # noqa: E402

NUM_LAYERS, TOPK, TOKENS = 4, 2, 3


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router_replay = RouterReplay()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.router = _Router()


def _chunk(layers: int, mtp: bool) -> torch.nn.Module:
    chunk = torch.nn.Module()
    chunk.decoder = torch.nn.Module()
    chunk.decoder.layers = torch.nn.ModuleList(_Layer() for _ in range(layers))
    if mtp:
        chunk.mtp = torch.nn.Module()
        chunk.mtp.layers = torch.nn.ModuleList([_Layer()])
    return chunk


def _replay(layer: torch.nn.Module) -> RouterReplay:
    return layer.mlp.router.router_replay


@pytest.fixture(autouse=True)
def _clean_registry():
    RouterReplay.global_router_replay_instances.clear()
    yield
    RouterReplay.global_router_replay_instances.clear()


def _setup(chunk: torch.nn.Module, layer_offset: int, layers: int) -> torch.Tensor:
    experts = torch.arange(TOKENS * NUM_LAYERS * TOPK).reshape(TOKENS, NUM_LAYERS * TOPK)
    cfg = SimpleNamespace(num_layers=NUM_LAYERS, moe_router_topk=TOPK, moe_layer_freq=1)
    with patch.object(model_mod, "get_model_config", return_value=cfg), \
         patch.object(model_mod, "prepare_routing_replay_data", side_effect=lambda seqs, _: seqs[0]), \
         patch.object(model_mod, "get_num_layers_to_build", return_value=layers), \
         patch.object(model_mod, "get_transformer_layer_offset", return_value=layer_offset):
        model_mod.setup_routing_replay(None, {"rollout_routed_experts": [experts]}, chunk, [chunk])
    return experts.reshape(TOKENS, NUM_LAYERS, TOPK)


def test_replaced_decoder_routers_do_not_shift_the_layer_mapping():
    # what Qwen3-VL's GPTModel.__init__ leaves behind: a stock decoder's routers, still registered
    _chunk(layers=2, mtp=False)
    # the last pipeline stage: layers 2 and 3 plus an MTP block
    chunk = _chunk(layers=2, mtp=True)
    packed = _setup(chunk, layer_offset=2, layers=2)

    for layer, layer_id in zip(chunk.decoder.layers, (2, 3)):
        assert torch.equal(_replay(layer).target_topk_idx, packed[:, layer_id])
        assert _replay(layer).router_replay_action == RouterReplayAction.REPLAY_FORWARD
    mtp = _replay(chunk.mtp.layers[0])
    assert mtp.target_topk_idx is None
    assert mtp.router_replay_action is None

    scores = torch.rand(TOKENS, 8)
    fallback = lambda s, k, *_, **__: torch.topk(s, k)  # noqa: E731
    probs, top = mtp.get_replay_topk(scores, TOPK, default_compute_topk=fallback)
    assert torch.equal(top, torch.topk(scores, TOPK).indices)


def test_backward_action_leaves_mtp_routing_itself():
    chunk = _chunk(layers=4, mtp=True)
    _setup(chunk, layer_offset=0, layers=4)
    model_mod.set_routing_replay_action([chunk], RouterReplayAction.REPLAY_BACKWARD)

    assert _replay(chunk.decoder.layers[0]).router_replay_action == RouterReplayAction.REPLAY_BACKWARD
    assert _replay(chunk.mtp.layers[0]).router_replay_action is None


def test_router_count_mismatch_raises():
    chunk = _chunk(layers=3, mtp=False)
    with pytest.raises(RuntimeError, match="3 decoder routers but 4 MoE layers"):
        _setup(chunk, layer_offset=0, layers=4)
