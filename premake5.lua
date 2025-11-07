-- Setup the extension
local ext = get_current_extension_info()
project_ext(ext)

-- (Optional) common Windows define to avoid min/max macro conflicts
defines { "NOMINMAX" }
language "C++"
cppdialect "C++17"

-- Build Python bindings loaded by the extension
project_ext_bindings {
    ext = ext,
    project_name = "uas.fbm",
    module = "_uas_fbm",
    src = "src/bindings",
    target_subdir = "uas/fbm"
}

-- ---------- Includes / Libs ----------
includedirs {
    "include",
    "src/cpp",
    "src/cuda"
}

add_usd { "usdGeom" }

-- ---------- Sources ----------
files {
    "extension.toml",
    "src/cpp/**.h", "src/cpp/**.hpp", "src/cpp/**.cpp",
    "src/cuda/**.cuh", "src/cuda/**.cu"
}

-- ---------- CUDA (Windows) ----------
filter { "system:windows" }
    local cuda_path = os.getenv("CUDA_PATH")
    if cuda_path ~= nil then
        libdirs { cuda_path .. "/lib/x64" }
        links { "cudart", "oldnames" }
    end
filter {}

-- NVCC custom compile for .cu files (single-TU flow)
filter { "files:src/cuda/**.cu" }
    buildmessage "NVCC %{file.relpath}"
    buildcommands {
        '"%CUDA_PATH%/bin/nvcc.exe" -ccbin "%VCToolsInstallDir%bin/Hostx64/x64" ' ..
        '-gencode=arch=compute_120,code=sm_120 --allow-unsupported-compiler ' ..
        '-Xcompiler "/MD" -c "%{file.relpath}" -o "%{cfg.objdir}/%{file.basename}_cuda.obj"'
    }
    buildoutputs { "%{cfg.objdir}/%{file.basename}_cuda.obj" }
    linkbuildoutputs "On"
filter {}

-- Debug uses /MDd so NVCC objects match the CRT of the rest of the project
filter { "configurations:Debug", "files:src/cuda/**.cu" }
    buildcommands {
        '"%CUDA_PATH%/bin/nvcc.exe" -ccbin "%VCToolsInstallDir%bin/Hostx64/x64" ' ..
        '-gencode=arch=compute_120,code=sm_120 --allow-unsupported-compiler ' ..
        '-Xcompiler "/MDd" -c "%{file.relpath}" -o "%{cfg.objdir}/%{file.basename}_cuda.obj"'
    }
filter {}

-- Stage assets/python into the built extension folder
repo_build.prebuild_link {
    { "data",         ext.target_dir .. "/data" },
    { "docs",         ext.target_dir .. "/docs" },
    { "python/impl",  ext.target_dir .. "/uas/fbm/impl" },
}

if os.isdir("python/tests") then
    repo_build.prebuild_link {
        { "python/tests", ext.target_dir .. "/uas/fbm/tests" }
    }
end
