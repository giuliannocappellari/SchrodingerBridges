"""Small frozen-base LoRA branch used by continual factual baselines."""

from __future__ import annotations

from typing import Any, Sequence


class LoRABranch:
    def __init__(self, model: Any, layers: Sequence[int], rank: int, alpha: float = 1.0):
        import torch

        from scripts.mdm_memit_editor import get_module, resolved_key_module_name

        self.torch = torch
        self.model = model
        self.layers = tuple(map(int, layers))
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.parameters_by_layer = {}
        self.handles = []
        for layer in self.layers:
            module = get_module(model, resolved_key_module_name(model, layer))
            output_width, input_width = map(int, module.weight.shape)
            a = torch.nn.Parameter(torch.empty(self.rank, input_width, device=module.weight.device, dtype=torch.float32))
            b = torch.nn.Parameter(torch.zeros(output_width, self.rank, device=module.weight.device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(a, a=5 ** 0.5)
            self.parameters_by_layer[layer] = (a, b)

            def make_hook(layer_index: int):
                def hook(_module, inputs, output):
                    values = inputs[0].float()
                    left, right = self.parameters_by_layer[layer_index]
                    residual = torch.nn.functional.linear(
                        torch.nn.functional.linear(values, left), right
                    ) * (self.alpha / self.rank)
                    return output + residual.to(output.dtype)
                return hook

            self.handles.append(module.register_forward_hook(make_hook(layer)))

    def parameters(self):
        for pair in self.parameters_by_layer.values():
            yield from pair

    def storage_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def state_dict_cpu(self):
        return {
            f"layer_{layer}_{name}": tensor.detach().cpu()
            for layer, pair in self.parameters_by_layer.items()
            for name, tensor in zip(("A", "B"), pair)
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
