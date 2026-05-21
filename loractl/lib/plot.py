import io
from PIL import Image
from modules import script_callbacks
import matplotlib
matplotlib.use('Agg')
import pandas as pd

log_weights = []
log_names = []
log_step = 0

def plot_lora_weight(weights, names):
    if not weights or not names:
        return Image.new('RGB', (1, 1))
    data = pd.DataFrame(weights)
    ax = data.plot()
    ax.set_xlabel("Steps")
    ax.set_ylabel("LoRA weight")
    ax.set_title("LoRA weight in all steps")
    ax.legend(loc=0)
    result_image = fig2img(ax)
    matplotlib.pyplot.close(ax.figure)
    del ax
    return result_image

def fig2img(fig):
    buf = io.BytesIO()
    fig.figure.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def reset_plot():
    global log_step
    log_weights.clear()
    log_names.clear()
    log_step = 0

def make_plot():
    return plot_lora_weight(log_weights, log_names)

def on_step(params):
    global log_step
    from loractl.lib.lora_ctl_network import lora_weights as raw_weights
    if not raw_weights:
        return

    current_step = log_step
    log_step += 1

    if not log_names:
        all_names = set()
        for w in raw_weights.values():
            all_names.update(w.keys())
        log_names.extend(sorted(all_names))

    from loractl.lib.forge_patching import _resolve_step_weight
    resolved = _resolve_step_weight(raw_weights, current_step) or {}

    frame = {}
    for name in log_names:
        frame[name] = float(resolved.get(name, 1.0))
    log_weights.append(frame)

script_callbacks.on_cfg_after_cfg(on_step)
