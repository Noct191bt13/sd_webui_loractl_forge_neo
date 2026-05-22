import os

import networks as forge_networks
from backend.args import dynamic_args
from backend.patcher.base import ModelPatcher as Patcher
from modules import script_callbacks, shared
from modules.script_callbacks import AfterCFGCallbackParams
from loractl.lib.lora_ctl_network import lora_weights, hr_lora_weights
from loractl.lib import utils

lora_patch_tracker = {}

_original_add_patches = None
_cfg_callback_count = 0
step_mode = False
_last_applied = None
_cached_name_to_file = None


def install_patch_tracker():
    global _original_add_patches
    if _original_add_patches is not None:
        return

    _original_add_patches = Patcher.add_patches

    def _tracking_add_patches(self, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        loaded_keys = _original_add_patches(self, patches, strength_patch, strength_model, filename=filename, online_mode=online_mode)
        if filename and online_mode:
            basename = os.path.splitext(os.path.basename(filename))[0]
            lora_patch_tracker.setdefault(basename, []).extend(loaded_keys)
        return loaded_keys

    Patcher.add_patches = _tracking_add_patches


def force_online_lora():
    dynamic_args.online_lora = True


def reset_cfg_counter():
    global _cfg_callback_count, _last_applied, _cached_name_to_file
    _cfg_callback_count = 0
    _last_applied = None
    _cached_name_to_file = None


def _build_name_to_filename():
    result = {}
    for net in forge_networks.loaded_networks:
        name = net.mentioned_name or net.name
        fname = os.path.splitext(os.path.basename(net.network_on_disk.filename))[0]
        result[name] = fname
    return result


def _resolve_step_weight(lora_weights, step):
    if not lora_weights:
        return None

    all_names = set()
    for w in lora_weights.values():
        all_names.update(w.keys())

    result = {}
    for name in all_names:
        steps_weights = {}
        for s, wdict in sorted(lora_weights.items()):
            if name in wdict:
                try:
                    steps_weights[s] = float(wdict[name])
                except (ValueError, TypeError):
                    steps_weights[s] = 1.0

        if not steps_weights:
            continue

        sorted_steps = sorted(steps_weights.keys())

        if step_mode:
            chosen = sorted_steps[0]
            for s in sorted_steps:
                if s <= step:
                    chosen = s
            result[name] = steps_weights[chosen]
        else:
            if step <= sorted_steps[0]:
                result[name] = steps_weights[sorted_steps[0]]
            elif step >= sorted_steps[-1]:
                result[name] = steps_weights[sorted_steps[-1]]
            else:
                for i in range(len(sorted_steps) - 1):
                    lo, hi = sorted_steps[i], sorted_steps[i + 1]
                    if lo <= step <= hi:
                        w_lo = steps_weights[lo]
                        w_hi = steps_weights[hi]
                        frac = (step - lo) / (hi - lo)
                        result[name] = w_lo + (w_hi - w_lo) * frac
                        break

    return result if result else None


def _update_online_patches(step_weight, unet, clip):
    for lora_name, target_weight in step_weight.items():
        for net in forge_networks.loaded_networks:
            name = net.mentioned_name or net.name
            if name == lora_name:
                fname = os.path.splitext(os.path.basename(net.network_on_disk.filename))[0]
                break
        else:
            fname = lora_name
        keys = lora_patch_tracker.get(fname, [])
        for key in keys:
            if key in unet.online_patches:
                patches = unet.online_patches[key]
                for i in range(len(patches)):
                    patches[i] = (float(target_weight),) + patches[i][1:]
            if clip is not None and hasattr(clip, 'patcher') and key in clip.patcher.online_patches:
                patches = clip.patcher.online_patches[key]
                for i in range(len(patches)):
                    patches[i] = (float(target_weight),) + patches[i][1:]


def _on_cfg_after_cfg(params: AfterCFGCallbackParams):
    global _cfg_callback_count, _last_applied
    if not utils.is_active():
        _cfg_callback_count = 0
        _last_applied = None
        return

    if not lora_weights:
        return

    objects = getattr(shared.sd_model, 'forge_objects', None)
    if objects is None:
        return
    unet = objects.unet
    if not unet.has_online_lora():
        return

    _cfg_callback_count += 1
    next_step = _cfg_callback_count

    if next_step >= params.total_sampling_steps:
        return

    weights_source = hr_lora_weights if utils.is_hires() and hr_lora_weights else lora_weights
    step_weight = _resolve_step_weight(weights_source, next_step)
    if step_weight is None:
        return

    if step_weight == _last_applied:
        return
    changed = step_weight if _last_applied is None else {k: v for k, v in step_weight.items() if _last_applied.get(k) != v}
    _last_applied = step_weight
    if shared.opts.data.get("loractl_log_weights", False):
        formatted = {k: round(v, 3) for k, v in changed.items()}
        print(f"[Loractl] Weights changed at step {next_step}: {formatted}")

    _update_online_patches(step_weight, unet, objects.clip)
    unet.refresh_loras()
    if objects.clip is not None and hasattr(objects.clip, 'patcher'):
        objects.clip.patcher.refresh_loras()


script_callbacks.on_cfg_after_cfg(_on_cfg_after_cfg)


def on_ui_settings():
    shared.opts.add_option("loractl_log_weights", shared.OptionInfo(False, "Log weight changes", section=("loractl", "Dynamic Lora Weights")))

script_callbacks.on_ui_settings(on_ui_settings)
