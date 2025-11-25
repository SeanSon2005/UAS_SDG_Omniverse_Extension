"""Compatibility shim forwarding to the canonical module location."""

import time
import threading
import http.server
import json

import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.viewport.utility
import omni.kit.app as app
from pxr import UsdGeom, Gf
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

from .. import _uas_fbm

CAMERA_OUTPUT_PATH = r"C:\Users\astro\Documents\Nvidia_Omniverse\kit-extension-template-cpp\captures\fbm_capture.png"

class TriggerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that schedules FBM generation on the main Kit thread."""

    def do_POST(self):
        if self.path == "/generate":
            try:
                # Schedule generation on the main Kit thread via the extension instance
                self.server.extension_instance.request_generate_and_capture()
                time.sleep(2.0)  # brief pause to ensure request is finished.
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", 
                                             "image_path": CAMERA_OUTPUT_PATH,
                                             "message": "Generation and capture finished."}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json") 
                self.end_headers()
                self.wfile.write(
                    json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                )
        else:
            self.send_response(404)
            self.end_headers()

    # Silence the default logging to stderr so it doesn't spam the Kit console
    def log_message(self, format, *args):  # noqa: A003 - keep name for override
        return


class FBM(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.label = None
        self.freq_model = ui.SimpleFloatModel(24.0)
        self.scale_model = ui.SimpleFloatModel(0.1)
        self.lacun_model = ui.SimpleFloatModel(2.0)
        self.persist_model = ui.SimpleFloatModel(0.5)
        self.seed_model = ui.SimpleIntModel(42)
        self.octaves_model = ui.SimpleIntModel(8)
        self.size_model = ui.SimpleIntModel(1024)
        self.mesh_scale_model = ui.SimpleFloatModel(2000.0)
        self.height_model = ui.SimpleFloatModel(200.0)
        self.mesh_prim_path = "/World/FBMTerrain"
        self.mesh = None

        self.http_server = None
        self.server_thread = None

        # Main-thread scheduling state
        self._pending_generate = False
        self._pending_capture = 0
        self._update_sub = None

    def on_startup(self, ext_id):
        print(f"FBM starting up (ext_id: {ext_id}).")

        self.camera_path = "/World/TopViewCamera"
        self.is_camera_created = False

        # Subscribe to Kit's main-thread update loop so we can safely call USD and UI code from HTTP requests.
        self._update_sub = (
            app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="uas.fbm_http_update")
        )

        # Start HTTP Server on a background thread
        self.start_http_server()

        # Build UI
        self._window = ui.Window("FBM Window", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                self.label = ui.Label("Ready")
                with ui.CollapsableFrame("Parameters", collapsed=False):
                    with ui.VStack(spacing=4, style={"padding": 4}):
                        self._build_float_row("Frequency", self.freq_model, 0.01, 32.0)
                        self._build_float_row("Scale", self.scale_model, 0.001, 1024.0)
                        self._build_float_row(
                            "Mesh Scale", self.mesh_scale_model, 0.001, 4096.0
                        )
                        self._build_float_row("Lacunarity", self.lacun_model, 0.01, 32.0)
                        self._build_float_row("Persistence", self.persist_model, 0.0, 1.0)
                        self._build_int_row("Init Seed", self.seed_model, -1_000_000, 1_000_000)
                        self._build_int_row("Octaves", self.octaves_model, 1, 16)
                        self._build_int_row("Size (px)", self.size_model, 1, 4096)
                        self._build_float_row(
                            "Height Scale", self.height_model, 0.0, 1_000_000.0
                        )
                ui.Button("Generate", clicked_fn=self.on_click)

    def start_http_server(self):
        try:
            server_address = ("localhost", 8211)
            self.http_server = http.server.HTTPServer(server_address, TriggerHandler)

            # Attach this extension instance so the handler can call request_generate()
            self.http_server.extension_instance = self

            self.server_thread = threading.Thread(
                target=self.http_server.serve_forever, name="FBM_HTTP_Server_Thread"
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            print("FBM HTTP Server started on port 8211")
        except Exception as e:
            print(f"Failed to start FBM HTTP Server: {e}")

    def _on_update(self, event):
        """
        Called on the main Kit thread every frame.
        Used for scheduling generate_and_capture calls from the HTTP thread.
        """
        if self._pending_generate:
            self._pending_generate = False
            self.on_click()
            self._pending_capture = 10
        
        elif self._pending_capture > 0:
            self._ensure_camera()
            self._pending_capture -= 1
            if self._pending_capture == 0:
                self._capture_camera()

    def request_generate_and_capture(self):
        """Thread-safe-ish generate and capture flag set from the HTTP thread.
        """
        self._pending_generate = True

    def on_shutdown(self):
        print("FBM shutting down.")

        # Stop HTTP server
        if self.http_server:
            try:
                self.http_server.shutdown()
                self.http_server.server_close()
            except Exception as e:
                print(f"Error shutting down HTTP server: {e}")
            self.http_server = None

        # Drop the update subscription (this unsubscribes in Kit)
        self._update_sub = None

    def on_click(self):
        """Generate the FBM mesh. Must run on the main Kit thread."""

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
            if self.label is not None:
                self.label.text = "Success [{:.5f} s]".format(time.time() - start_time)
        except Exception as exc:
            print(f"FBM Terrain generation failed: {exc}")
            if self.label is not None:
                self.label.text = "Fail"

    def _ensure_camera(self):
        """Ensure the top-down camera exists and is set as active in the viewport."""
        if not self.is_camera_created:
            try:
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    raise RuntimeError("No active USD stage.")
                camera: UsdGeom.Camera = UsdGeom.Camera.Define(stage, self.camera_path)
                xform_api = UsdGeom.XformCommonAPI(camera.GetPrim())
                xform_api.SetTranslate(Gf.Vec3d(0.0, 2500.0, 0.0))
                xform_api.SetRotate(Gf.Vec3f(270.0, 0.0, 0.0))
                xform_api.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
                self.is_camera_created = True
                viewport = get_active_viewport()
                viewport.set_active_camera(self.camera_path)
            except Exception as e:
                print(f"Failed to create camera: {e}")

    def _capture_camera(self):
        """Capture the active viewport to a file."""
        capture_viewport_to_file(get_active_viewport(), CAMERA_OUTPUT_PATH)

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
