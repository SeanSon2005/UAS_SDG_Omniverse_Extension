import asyncio
import base64
import os
import subprocess
import sys
import time
import requests
import logging
import threading

# Configure logging to stderr so it doesn't interfere with stdout (MCP protocol)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("omniverse-sdg")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Initialize Server
app = Server("OmniverseSDG")

# Configuration
OMNIVERSE_PATH = "C:\\Users\\astro\\Documents\\Nvidia_Omniverse\\kit-extension-template-cpp"
OMNIVERSE_MCP_PATH = os.path.join(OMNIVERSE_PATH, "source", "extensions", "uas.fbm", "mcp_server")
BUILD_SCRIPT = "build.bat"
LAUNCH_SCRIPT = os.path.join("_build", "windows-x86_64", "release", "omni.app.kit.dev.bat")
EXTENSION_API_URL = "http://localhost:8211/generate"
LOG_PATH = os.path.join(OMNIVERSE_MCP_PATH, "build_output.log")

# Global Omniverse Process
OMNIVERSE_PROC: subprocess.Popen | None = None

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="verify_build_environment",
             description="Verifies that the Omniverse build environment is correctly set up.",
             inputSchema={
                 "type": "object",
                 "properties": {},
             },
        ),
        Tool(
            name="build_extension",
            description="Builds the Omniverse extension using the build.bat script.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="launch_omniverse",
            description="Launches Omniverse using the omni.app.kit.dev.bat script.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="close_omniverse",
            description="Closes the running Omniverse application.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="generate_data",
            description="Triggers the data generation in the running Omniverse extension.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    global OMNIVERSE_PROC

    if name == "verify_build_environment":
        check_nvcc = subprocess.run("nvcc --version", capture_output=True, text=True)
        check_cl = subprocess.run("where cl", capture_output=True, text=True)
        check_env = os.environ.copy()
        msg = ""
        okay_count = 0
        if check_nvcc.returncode != 0:
            msg += "NVIDIA CUDA Compiler (nvcc) not found in PATH.\n"
        else:
            msg += "NVIDIA CUDA Compiler (nvcc) found:\n" + check_nvcc.stdout + "\n"
            okay_count += 1
        if check_cl.returncode != 0:
            msg += "Microsoft C++ Compiler (cl) not found in PATH.\n"
        else:
            msg += "Microsoft C++ Compiler (cl) found at:\n" + check_cl.stdout + "\n"
            okay_count += 1
        if check_env.get("PATHEXT") is None:
            msg += "PATHEXT environment variable is not set.\n"
        else:
            msg += "PATHEXT environment variable is set: " + check_env.get("PATHEXT") + "\n"
            okay_count += 1
        if okay_count >= 3:
            msg = "Build environment verification successful:\n\n" + msg
        else:
            msg = "Build environment verification failed:\n\n" + msg
        return [TextContent(type="text", text=msg)]
        
    elif name == "build_extension":
        try:
            logger.info("Starting build process using script: %s", BUILD_SCRIPT)

            # Start the build process
            result = subprocess.run(f"{BUILD_SCRIPT}", 
                                    cwd=OMNIVERSE_PATH, 
                                    shell=True, 
                                    text=True, 
                                    stderr=subprocess.PIPE, 
                                    stdout=subprocess.PIPE, 
                                    stdin=subprocess.DEVNULL)
            
            if result.returncode != 0:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Build FAILED (exit code {result.returncode}).\n\n"
                            f"STDOUT:\n{result.stdout}\n\n"
                            f"STDERR:\n{result.stderr}"
                        ),
                    )
                ]

            return [
                TextContent(
                    type="text",
                    text=f"Build successful:\n{result.stdout}",
                )
            ]

        except Exception as e:
            logger.exception("Unexpected error during build")
            return [TextContent(
                type="text",
                text=f"An error occurred during build: {str(e)}",
            )]

    elif name == "launch_omniverse":
        try:
            # Start the process
            proc = subprocess.Popen(
                LAUNCH_SCRIPT,
                cwd=OMNIVERSE_PATH,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # keep this as requested
            )
            OMNIVERSE_PROC = proc

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            # Reader helper that runs in a background thread and accumulates lines
            def _read_stream(stream, sink):
                try:
                    for line in iter(stream.readline, ""):
                        sink.append(line)
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            # Start background readers so the main thread can do a timed wait
            t_out = threading.Thread(
                target=_read_stream, args=(proc.stdout, stdout_lines), daemon=True
            )
            t_err = threading.Thread(
                target=_read_stream, args=(proc.stderr, stderr_lines), daemon=True
            )
            t_out.start()
            t_err.start()

            start = time.time()
            app_ready = False

            # Poll for "app ready" or timeout / early exit
            while True:
                # Check for the sentinel text for app readiness
                if any("app ready" in line for line in stdout_lines):
                    app_ready = True
                    break
                # If the process exited on its own, stop waiting
                if proc.poll() is not None:
                    break

                # Timeout
                if time.time() - start > 15:
                    break
                time.sleep(0.1)

            # Give the reader threads a brief moment to flush any last lines
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
            stdout_text = "".join(stdout_lines)
            stderr_text = "".join(stderr_lines)

            # App Ready Found
            if app_ready:
                return [
                    TextContent(
                        type="text",
                        text=(
                            "Omniverse reported 'app ready' within 15 seconds.\n\n"
                            "STDOUT (captured):\n"
                            f"{stdout_text}"
                        ),
                    )
                ]
            
            # App not ready, handle termination
            timed_out = time.time() - start > 15
            if proc.poll() is None:
                try:
                    proc.terminate()  # sends TerminateProcess on Windows, no Y/N prompt
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                        
            if timed_out:
                reason = "Timeout waiting for Omniverse to Launch (15s). Process terminated."
            else:
                # Process exited before printing 'app ready'
                code = proc.returncode
                reason = f"Process exited with code {code} before Omniverse reported 'app ready'"

            return [
                TextContent(
                    type="text",
                    text=(
                        f"{reason}.\n\n"
                        f"STDOUT (captured):\n{stdout_text}\n\n"
                        f"STDERR (captured):\n{stderr_text}"
                    ),
                )
            ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Failed to launch Omniverse: {type(e).__name__}: {e}",
                )
            ]
        
    elif name == "close_omniverse":
        try:
            # Check if OMNIVERSE Process is Running
            if OMNIVERSE_PROC is not None:
                proc = OMNIVERSE_PROC
                OMNIVERSE_PROC = None # Clear global reference

                if proc.poll() is None:
                    try:
                        proc.kill()
                        subprocess.run(["taskkill", "/IM", "kit.exe", "/F"], 
                                       check=True, 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL)
                        return [TextContent(
                            type="text",
                            text="Omniverse Shutdown Complete.",
                        )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=f"Error while terminating tracked Omniverse process: {type(e).__name__}: {e}",
                        )]
                else:
                    return [TextContent(
                        type="text",
                        text="Tracked Omniverse process had already exited. No action taken.",
                    )]
            else:
                return [TextContent(
                    type="text",
                    text="No tracked Omniverse process. No action taken.",
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Failed to close Omniverse: {type(e).__name__}: {e}",
            )]


    elif name == "generate_data":
        try:
            response = await asyncio.to_thread(requests.post, EXTENSION_API_URL, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data.get('status') == 'success':
                image_path = data.get("image_path")
                if image_path and os.path.exists(image_path):

                    # Read image bytes and encode to base64
                    with open(image_path, "rb") as f:
                        img_bytes = f.read()
                    b64_data = base64.b64encode(img_bytes).decode("ascii")

                    return [
                        TextContent(
                            type="text",
                            text=f"Generation successful. Here's the captured image from Omniverse's viewport:"
                        ),
                        ImageContent(
                            type="image",
                            data=b64_data,
                            mimeType="image/png",
                        ),
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text="Generation reported success, but image file not found."
                        )
                    ]
            else:
                return [TextContent(type="text", text=f"Generation failed: {data.get('message')}")]

        except requests.exceptions.ConnectionError:
            return [TextContent(type="text", text="Failed to connect to Omniverse extension. Launch Omniverse and try again.")]
        except Exception as e:
            return [TextContent(type="text", text=f"An error occurred during generation: {str(e)}")]

    raise ValueError(f"Tool not found: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
