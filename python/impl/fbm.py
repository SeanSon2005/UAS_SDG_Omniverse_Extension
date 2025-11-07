"""Compatibility shim forwarding to the canonical module location."""

import os
import time
from typing import List, Dict

import omni.ext
import omni.ui as ui
import omni.usd
from pxr import UsdGeom, Sdf, Vt, Gf, Usd

from .. import _uas_fbm


_LAYER_DEFS = [
    {
        "label": "1) Hard Court Light Green",
        "usd_name": "Ground_Hard_Court_Light_Green",
        "diffuse": "Ground_Hard_Court_Light_Green_diffuse.jpg",
        "normal": "Ground_Hard_Court_Light_Green_normal.jpg",
        "roughness": "Ground_Hard_Court_Light_Green_glossy_roughness.jpg",
        "specular": "Ground_Hard_Court_Light_Green_glossy_weight.jpg",
        "default_range": (0.85, 1.0),
        "default_blend": 1.0,
        "specular_constant": 0.6,
        "roughness_constant": 0.4,
    },
    {
        "label": "2) Hard Court Olive Green",
        "usd_name": "Ground_Hard_Court_Olive_Green",
        "diffuse": "Ground_Hard_Court_Olive_Green_diffuse.jpg",
        "normal": "Ground_Hard_Court_Olive_Green_normal.jpg",
        "roughness": "Ground_Hard_Court_Olive_Green_glossy_roughness.jpg",
        "specular": "Ground_Hard_Court_Olive_Green_glossy_weight.jpg",
        "default_range": (0.75, 0.92),
        "default_blend": 0.9,
        "specular_constant": 0.55,
        "roughness_constant": 0.45,
    },
    {
        "label": "3) Agg. Saturated Moss Patches",
        "usd_name": "Ground_Aggregate_Saturated_Moss_Patches",
        "diffuse": "Ground_Aggregate_Saturated_Moss_Patches_diffuse.jpg",
        "normal": "Ground_Aggregate_Saturated_Moss_Patches_normal.jpg",
        "roughness": "Ground_Aggregate_Saturated_Moss_Patches_glossy_roughness.jpg",
        "specular": "",
        "default_range": (0.6, 0.85),
        "default_blend": 0.85,
        "specular_constant": 0.35,
        "roughness_constant": 0.6,
    },
    {
        "label": "4) Agg. Exposed",
        "usd_name": "Ground_Aggregate_Exposed",
        "diffuse": "Ground_Aggregate_Exposed_diffuse.jpg",
        "normal": "Ground_Aggregate_Exposed_normal.jpg",
        "roughness": "Ground_Aggregate_Exposed_glossy_roughness.jpg",
        "specular": "",
        "default_range": (0.45, 0.65),
        "default_blend": 0.8,
        "specular_constant": 0.4,
        "roughness_constant": 0.55,
    },
    {
        "label": "5) Mulch",
        "usd_name": "Mulch",
        "diffuse": "Mulch_diffuse.jpg",
        "normal": "Mulch_normal.jpg",
        "roughness": "Mulch_glossy_roughness.jpg",
        "specular": "Mulch_glossy_weight.jpg",
        "default_range": (0.3, 0.55),
        "default_blend": 0.75,
        "specular_constant": 0.3,
        "roughness_constant": 0.65,
    },
    {
        "label": "6) Mulch Wet",
        "usd_name": "Mulch_Wet",
        "diffuse": "Mulch_Wet_diffuse.jpg",
        "normal": "Mulch_Wet_normal.jpg",
        "roughness": "Mulch_Wet_glossy_roughness.jpg",
        "specular": "Mulch_Wet_glossy_weight.jpg",
        "default_range": (0.15, 0.35),
        "default_blend": 0.7,
        "specular_constant": 0.4,
        "roughness_constant": 0.55,
    },
    {
        "label": "7) Small Gravel Rough",
        "usd_name": "Small_Gravel_Rough",
        "diffuse": "Small_Gravel_Rough_diffuse.jpg",
        "normal": "Small_Gravel_Rough_normal.jpg",
        "roughness": "Small_Gravel_Rough_glossy_roughness.jpg",
        "specular": "Small_Gravel_Rough_glossy_weight.jpg",
        "default_range": (0.0, 0.25),
        "default_blend": 0.65,
        "specular_constant": 0.5,
        "roughness_constant": 0.6,
    },
]

_BAKE_OUTPUT_STEMS = {
    "base_color": "fbm_base_color",
    "normal": "fbm_normal",
    "roughness": "fbm_roughness",
    "specular": "fbm_specular",
}


class FBM(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.label = None
        impl_dir = os.path.dirname(__file__)
        self._extension_root = os.path.abspath(os.path.join(impl_dir, "..", "..", ".."))
        self.freq_model = ui.SimpleFloatModel(16.0)
        self.scale_model = ui.SimpleFloatModel(0.2)
        self.lacun_model = ui.SimpleFloatModel(2.0)
        self.persist_model = ui.SimpleFloatModel(0.5)
        self.seed_model = ui.SimpleIntModel(42)
        self.octaves_model = ui.SimpleIntModel(8)
        self.size_model = ui.SimpleIntModel(4096)
        self.mesh_scale_model = ui.SimpleFloatModel(1000.0)
        self.height_model = ui.SimpleFloatModel(50.0)
        self.mesh_prim_path = "/World/FBMTerrain"
        self.mesh = None
        self._bake_status_label = None
        self.layer_models = []
        for layer in _LAYER_DEFS:
            self.layer_models.append(
                {
                    "config": layer,
                    "min_model": ui.SimpleFloatModel(layer["default_range"][0]),
                    "max_model": ui.SimpleFloatModel(layer["default_range"][1]),
                    "blend_model": ui.SimpleFloatModel(layer["default_blend"]),
                }
            )

    def on_startup(self, ext_id):
        print(f"FBM starting up (ext_id: {ext_id}).")

        self._window = ui.Window("FBM Window", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                self.label = ui.Label("Ready")
                with ui.CollapsableFrame("Parameters", collapsed=False):
                    with ui.VStack(spacing=4, style={"padding": 4}):
                        self._build_float_row("Frequency", self.freq_model, 0.01, 32.0)
                        self._build_float_row("Scale", self.scale_model, 0.001, 1024.0)
                        self._build_float_row("Mesh Scale", self.mesh_scale_model, 0.001, 4_096.0)
                        self._build_float_row("Lacunarity", self.lacun_model, 0.01, 32.0)
                        self._build_float_row("Persistence", self.persist_model, 0.0, 1.0)
                        self._build_int_row("Init Seed", self.seed_model, -1_000_000, 1_000_000)
                        self._build_int_row("Octaves", self.octaves_model, 1, 16)
                        self._build_int_row("Size (px)", self.size_model, 1, 4_096)
                        self._build_float_row("Height Scale", self.height_model, 0.0, 1_000_000.0)
                with ui.CollapsableFrame("Texture Bake", collapsed=False):
                    with ui.VStack(spacing=4, style={"padding": 4}):
                        self._bake_status_label = ui.Label(
                            "Baked textures saved under data/materials/output/...",
                            word_wrap=True,
                            height=30,
                        )
                        for layer in self.layer_models:
                            self._build_layer_controls(layer)
                ui.Button("Generate", clicked_fn=self.on_click)

    def on_shutdown(self):
        print(f"FBM shutting down.")

    def on_click(self):
        try:
            start_time = time.time()
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("No active USD stage.")

            size = max(1, int(self.size_model.get_value_as_int()))
            freq = float(self.freq_model.get_value_as_float())
            scale = float(self.scale_model.get_value_as_float())
            mesh_scale = float(self.mesh_scale_model.get_value_as_float())
            lacun = float(self.lacun_model.get_value_as_float())
            persist = float(self.persist_model.get_value_as_float())
            seed = int(self.seed_model.get_value_as_int())
            octaves = max(1, int(self.octaves_model.get_value_as_int()))
            height_scale = float(self.height_model.get_value_as_float())
            layer_settings = self._collect_material_layer_settings()
            texture_output_paths = self._resolve_texture_output_paths()

            _uas_fbm.generate_fbm_mesh(
                stage,
                size,
                scale,
                freq,
                mesh_scale,
                lacun,
                persist,
                seed,
                octaves,
                height_scale,
                self.mesh_prim_path,
                layer_settings,
                texture_output_paths,
            )

            self.mesh = self.mesh_prim_path
            self._update_bake_status_label(texture_output_paths)
            self.label.text = "Success [{:.5f} s]".format(time.time() - start_time)
        except Exception as exc:
            print(f"FBM Terrain generation failed: {exc}")
            self.label.text = "Fail"

    def _build_float_row(self, title, model, minimum, maximum):
        """Create a labeled float input row."""
        with ui.HStack(height=24):
            ui.Label(title, width=90, alignment=ui.Alignment.LEFT_CENTER)
            ui.FloatDrag(model=model, min=minimum, max=maximum, step=0.05, width=140)

    def _build_int_row(self, title, model, minimum, maximum):
        """Create a labeled integer input row."""
        with ui.HStack(height=24):
            ui.Label(title, width=90, alignment=ui.Alignment.LEFT_CENTER)
            ui.IntDrag(model=model, min=minimum, max=maximum, step=1, width=140)

    def _build_layer_controls(self, layer_entry):
        config = layer_entry["config"]
        ui.Label(config["label"], height=18)
        self._build_float_row("Height Min", layer_entry["min_model"], 0.0, 1.0)
        self._build_float_row("Height Max", layer_entry["max_model"], 0.0, 1.0)
        self._build_float_row("Blend", layer_entry["blend_model"], 0.0, 1.0)

    def _collect_material_layer_settings(self) -> List[Dict[str, float]]:
        layers = []
        for layer in self.layer_models:
            config = layer["config"]
            height_min = float(layer["min_model"].get_value_as_float())
            height_max = float(layer["max_model"].get_value_as_float())
            blend_amount = float(layer["blend_model"].get_value_as_float())
            if height_max < height_min:
                height_min, height_max = height_max, height_min

            texture_path = self._resolve_material_texture(config.get("diffuse"))
            layers.append(
                {
                    "usd_name": config["usd_name"],
                    "diffuse_texture": texture_path,
                    "normal_texture": self._resolve_material_texture(config.get("normal")),
                    "roughness_texture": self._resolve_material_texture(config.get("roughness")),
                    "specular_texture": self._resolve_material_texture(config.get("specular")),
                    "height_min": max(0.0, min(1.0, height_min)),
                    "height_max": max(0.0, min(1.0, height_max)),
                    "blend_amount": max(0.0, min(1.0, blend_amount)),
                    "specular_constant": float(config.get("specular_constant", 0.5)),
                    "roughness_constant": float(config.get("roughness_constant", 0.5)),
                }
            )
        return layers

    def _resolve_material_texture(self, filename: str) -> str:
        if not filename:
            return ""
        texture_path = os.path.join(
            self._extension_root, "data", "materials", "textures", filename
        )
        return os.path.abspath(texture_path)

    def _resolve_texture_output_paths(self) -> Dict[str, str]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time() * 1000) % 1000
        suffix = f"{timestamp}_{millis:03d}"
        base_dir = os.path.join(self._extension_root, "data", "materials", "output")
        outputs = {}
        for key, stem in _BAKE_OUTPUT_STEMS.items():
            filename = f"{stem}_{suffix}.png"
            outputs[key] = os.path.abspath(os.path.join(base_dir, filename))
        return outputs

    def _update_bake_status_label(self, outputs: Dict[str, str]):
        if not self._bake_status_label:
            return
        base_preview = os.path.basename(outputs.get("base_color", "")) if outputs else ""
        if base_preview:
            self._bake_status_label.text = f"Baked textures: {base_preview} (and matching normals/rough/spec)"
        else:
            self._bake_status_label.text = "Baked textures saved under data/materials/output/"
