"""Compatibility shim forwarding to the canonical module location."""

import omni.ext
import time
import omni.ui as ui
import omni.usd
from pxr import UsdGeom, Sdf, Vt, Gf, Usd

from .. import _uas_fbm


class FBM(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.label = None
        self.freq_model = ui.SimpleFloatModel(16.0)
        self.scale_model = ui.SimpleFloatModel(0.1)
        self.lacun_model = ui.SimpleFloatModel(2.0)
        self.persist_model = ui.SimpleFloatModel(0.5)
        self.seed_model = ui.SimpleIntModel(42)
        self.octaves_model = ui.SimpleIntModel(8)
        self.size_model = ui.SimpleIntModel(1024)
        self.mesh_scale_model = ui.SimpleFloatModel(2000.0)
        self.height_model = ui.SimpleFloatModel(150.0)
        self.mesh_prim_path = "/World/FBMTerrain"
        self.mesh = None

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
            )

            self.mesh = self.mesh_prim_path
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
