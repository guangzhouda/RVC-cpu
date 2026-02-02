$ErrorActionPreference = "Stop"

# Build sdk/rvc_sdk_ort using local deps/onnxruntime + deps/faiss
# Note: For Visual Studio generator you should run this in:
#   "x64 Native Tools Command Prompt for VS 2022" (or Developer PowerShell),
# so that cl.exe is available.

$ortRoot = "deps/onnxruntime/onnxruntime-win-x64-gpu-1.17.1"
$faissRoot = "deps/faiss"

cmake -S sdk/rvc_sdk_ort -B build_rvc_sdk_ort -G "Visual Studio 17 2022" -A x64 `
  -DONNXRUNTIME_ROOT="$ortRoot" `
  -DFAISS_ROOT="$faissRoot"

cmake --build build_rvc_sdk_ort --config Release

# Stage runtime DLLs next to the demo exe for quick testing.
$demoDir = "build_rvc_sdk_ort/Release"
if (Test-Path $demoDir) {
  Copy-Item -Force -Path "$ortRoot/lib/onnxruntime.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$ortRoot/lib/onnxruntime_providers_cuda.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/faiss.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/libblas.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/liblapack.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Write-Host "Staged runtime DLLs into: $demoDir"
}
