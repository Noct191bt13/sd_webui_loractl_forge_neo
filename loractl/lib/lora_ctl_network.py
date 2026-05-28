from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils

import sys
from pathlib import Path

lora_parent = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin")
sys.path.insert(0, lora_parent)
from sd_forge_lora import network, networks, extra_networks_lora  # type: ignore
sys.path.remove(lora_parent)

from modules.processing import StableDiffusionProcessing
from modules.extra_networks import ExtraNetworkParams

lora_weights = {}
hr_lora_weights = {}

params_map = {}

def reset_weights():
    lora_weights.clear()
    hr_lora_weights.clear()
    params_map.clear()

class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    def __init__(self):
        self.params_list = []
        super().__init__()

    def clear(self):
        self.params_list = []
        reset_weights()

    def activate(self, p: StableDiffusionProcessing, params_list: list[ExtraNetworkParams]):
        if not utils.is_active():
            return super().activate(p, params_list)

        self.params_list = params_list
        for params in params_list:
            assert params.items
            name = params.positional[0]
            initial_weight = 1.0
            if len(params.positional) > 1:
                try:
                    weight_str = str(params.positional[1])
                    if '@' not in weight_str and ',' not in weight_str:
                        initial_weight = float(params.positional[1])
                except ValueError:
                    pass
            weights = utils.sorted_positions(params.positional[1], p.steps)
            for start_step, value in weights.items():
                if start_step not in lora_weights:
                    lora_weights[start_step] = {}
                lora_weights[start_step][name] = value
            hr_raw = params.named.get("hr")
            if hr_raw is not None:
                hr_weights = utils.sorted_positions(str(hr_raw), p.steps)
                for start_step, value in hr_weights.items():
                    if start_step not in hr_lora_weights:
                        hr_lora_weights[start_step] = {}
                    hr_lora_weights[start_step][name] = value
            params.positional = [name, initial_weight]
            params.named = {}
            params_map[name] = params

        for step in sorted(lora_weights, reverse=True):
            for (name, weight) in lora_weights[step].items():
                params_map[name].positional = [name, str(weight)]
                params_map[name].named = {}
        super().activate(p, self.params_list)
