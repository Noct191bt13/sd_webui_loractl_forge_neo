def patch_xyz_grid():
    from modules import scripts
    xyz_grid = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid") and hasattr(data, "module"):
            xyz_grid = data.module
            break
    if xyz_grid is None:
        return

    label = "[LoraCtl] "
    if any(x.label == label for x in xyz_grid.axis_options):
        return

    def apply_weight_mode(p, x, _xs):
        import loractl.lib.forge_patching as fp
        from loractl.lib import utils
        global _lora_ctl_xyz_enabled
        if x == "Dynamic":
            fp.step_mode = False
        elif x == "Static":
            fp.step_mode = True
        else:
            return
        _lora_ctl_xyz_enabled = True
        utils.set_active(True)

    def choices():
        return ["Dynamic", "Static"]

    xyz_grid.axis_options.append(
        xyz_grid.AxisOption(label + "Weight Mode", str, apply_weight_mode, choices=choices)
    )


_lora_ctl_xyz_enabled = False

try:
    patch_xyz_grid()
except Exception:
    pass
