import modules.scripts as scripts
from modules import extra_networks
from modules.processing import StableDiffusionProcessing
import gradio as gr
from loractl.lib import utils, plot, lora_ctl_network, forge_patching, xyz_integration

class LoraCtlScript(scripts.Script):
    def __init__(self):
        self.original_network = None
        self.network_replaced = False
        forge_patching.install_patch_tracker()
        super().__init__()

    sorting_priority = 10.2

    def title(self):
        return "Dynamic Lora Weights (reForge)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Dynamic Lora Weights", open=False):
                opt_enable = gr.Checkbox(
                    value=True, label="Enable", elem_id="loractl_enable")
                opt_weight_mode = gr.Radio(
                    choices=["Static", "Dynamic"],
                    value="Static",
                    label="Weight mode",
                    elem_id="loractl_weight_mode",
                    info="Dynamic = smooth interpolation, Static = instant jumps at scheduled steps")
                opt_plot_lora_weight = gr.Checkbox(
                    value=False, label="Plot the LoRA weight in all steps", elem_id="loractl_plot")
        return [opt_enable, opt_weight_mode, opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, opt_enable=True, opt_weight_mode="Static", opt_plot_lora_weight=False, **kwargs):
        current_network = extra_networks.extra_network_registry.get("lora")
        if opt_enable and not isinstance(current_network, lora_ctl_network.LoraCtlNetwork):
            self.original_network = current_network
            network = lora_ctl_network.LoraCtlNetwork()
            extra_networks.register_extra_network(network)
            extra_networks.register_extra_network_alias(network, "loractl")
            self.network_replaced = True
        elif not opt_enable and self.network_replaced:
            extra_networks.register_extra_network(self.original_network)
            self.original_network = None
            self.network_replaced = False
            try:
                from modules import sd_models
                current_sd = sd_models.model_data.get_sd_model()
                if current_sd is not None:
                    current_sd.current_lora_hash = ''
            except Exception:
                pass

        utils.set_hires(False)
        utils.set_active(opt_enable)
        forge_patching.step_mode = (opt_weight_mode == "Static") and opt_enable
        if opt_enable:
            forge_patching.force_online_lora()
        lora_ctl_network.reset_weights()
        forge_patching.reset_cfg_counter()
        plot.reset_plot()

    def before_hr(self, p, *args):
        utils.set_hires(True)
        if utils.is_active():
            forge_patching.reset_cfg_counter()

    def postprocess(self, p, processed, opt_enable=True, opt_weight_mode="Dynamic", opt_plot_lora_weight=False, **kwargs):
        if opt_plot_lora_weight and opt_enable:
            processed.images.extend([plot.make_plot()])
