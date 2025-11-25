const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const serverDir = __dirname;
const serverScript = path.join(serverDir, "server.py");

if (!fs.existsSync(serverScript)) {
  console.error("[wrapper] Unable to locate server.py at", serverScript);
  process.exit(1);
}

const pythonCommand = "C:\\Users\\astro\\miniconda3\\python.exe"
const vsDevCmd = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\Tools\\VsDevCmd.bat";
const launchCmd = [
  `call "${vsDevCmd}" -arch=x64 -host_arch=x64 > nul 2>&1`,
  `cd /d "${serverDir}"`,
  `${pythonCommand} "${serverScript}"`,
].join(" && ");

// Make sure essential environment variables are set
const env = { ...process.env };
if (!env.PROCESSOR_ARCHITECTURE) env.PROCESSOR_ARCHITECTURE = "AMD64";
if (!env.OS) env.OS = "Windows_NT";
if (!env.SystemRoot && !env.SYSTEMROOT) env.SystemRoot = "C:\\Windows";
if (!env.windir) env.windir = "C:\\Windows";
if (!env.COMSPEC && !env.ComSpec) env.COMSPEC = "C:\\Windows\\system32\\cmd.exe";
if (!env.TEMP) env.TEMP = "C:\\Users\\astro\\AppData\\Local\\Temp";
if (!env.TMP) env.TMP = env.TEMP;
if (!env.CommonProgramFiles) env.CommonProgramFiles = "C:\\Program Files\\Common Files";
if (!env["CommonProgramFiles(x86)"]) env["CommonProgramFiles(x86)"] = "C:\\Program Files (x86)\\Common Files";
if (!env.CommonProgramW6432) env.CommonProgramW6432 = "C:\\Program Files\\Common Files";
if (!env.CUDA_PATH) env.CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0";
if (!env.CUDA_PATH_V13_0) env.CUDA_PATH_V13_0 = env.CUDA_PATH;

const child = spawn(launchCmd, {
  cwd: serverDir,
  stdio: "inherit",
  env: env,
  shell: true,
});
const forwardSignal = (signal) => {
  if (child.killed) {
    return;
  }
  try {
    child.kill(signal);
  } catch (err) {
    // Ignore errors when forwarding signals on Windows
  }
};
process.on("SIGINT", () => forwardSignal("SIGINT"));
process.on("SIGTERM", () => forwardSignal("SIGTERM"));
child.on("exit", (code, signal) => {
  console.error(`\n[wrapper] MCP server exited code=${code}, signal=${signal}`);
  process.exitCode = code ?? 1;
});
