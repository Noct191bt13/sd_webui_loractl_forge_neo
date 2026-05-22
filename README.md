# LoRa Control — Dynamic Weights (Forge Neo)

Fork of [Panchovix/sd_webui_loractl_reforge_y](https://github.com/Panchovix/sd_webui_loractl_reforge_y), itself a fork of [cheald/sd-webui-loractl](https://github.com/cheald/sd-webui-loractl). Adapted for SD WebUI Forge Neo's patch-based LoRA system.

Dynamically control LoRA weights over the course of generation using keyframe syntax.

## Features

- Keyframe weight schedules with `weight@step` syntax
- **Dynamic mode** (default): smooth interpolation between keyframes
- **Static mode**: instant weight jumps at scheduled steps (no ramp)
- Per-step weight plot
- Separate high-res pass schedules
- XYZ grid integration — grid over Dynamic/Static modes

## Installation

In the webui Extensions tab → "Install from URL":

```
https://github.com/Noct191bt13/sd_webui_loractl_forge_neo
```

Then "Apply and restart UI".

## Usage

Enable **Dynamic Lora Weights** from the accordion in the txt2img/img2img panel.

### Weight schedule syntax

Standard LoRA: `<lora:name:weight>`

Dynamic schedule: `<lora:name:weight1@step1,weight2@step2,...>`

Steps `< 1.0` are fractions of total steps (e.g. `0.5` = 50%).  
Steps `>= 1.0` are absolute step numbers.

Examples:

- `<lora:mix:0@0,1@0.5,0@1>` — ramp up then back down
- `<lora:detail:0.1@0,1@1>` — grow from 0.1 to full over entire generation
- `<lora:style:0@0,1@10>` — off until step 10, then on

### Weight mode

| Mode | Behavior |
|------|----------|
| **Static** (default) | Instant jump at each keyframe, held constant between them |
| **Dynamic** | Smooth interpolation between keyframes |

### Separate high-res pass

```
<lora:name:0@0,1@1:hr=0.5@0,1@1>
```

Named params: `hr`

### XYZ grid

Select `[LoraCtl] Weight Mode` as an X/Y/Z axis with values `Dynamic` / `Static`.

## How it works

Forge Neo loads LoRAs as online patches (`ModelPatcher.online_patches`). This extension hooks `cfg_after_cfg` and updates the patch strengths in-place before each sampling step. `refresh_loras()` propagates changes to the model layers before the next forward pass.

## Credits

- [cheald](https://github.com/cheald) — original sd-webui-loractl
- [Panchovix](https://github.com/Panchovix) — reForge port
