$ErrorActionPreference = "Stop"

# Build sdk/rvc_sdk_ort using local deps/onnxruntime + deps/faiss
# Note: For Visual Studio generator you should run this in:
#   "x64 Native Tools Command Prompt for VS 2022" (or Developer PowerShell),
# so that cl.exe is available.

$ortRoot = "deps/onnxruntime/onnxruntime-win-x64-gpu-1.17.1"
$faissRoot = "deps/faiss"
$buildDir = "build_rvc_sdk_ort"

# 说明：如果项目目录被移动/重命名（例如从 Retrieval-based-Voice-Conversion-WebUI 改成 RVC-cpu），
# 旧的 CMakeCache 会记录旧路径，导致配置失败。这里做一次自动清理。
$cache = Join-Path $buildDir "CMakeCache.txt"
if (Test-Path $cache) {
  $cachedHomeLine = Select-String -Path $cache -Pattern "CMAKE_HOME_DIRECTORY:INTERNAL=" -SimpleMatch | Select-Object -First 1
  if ($cachedHomeLine) {
    $cachedHome = ($cachedHomeLine.Line -split "=", 2)[1].Trim()
    $expectedHome = (Resolve-Path "sdk/rvc_sdk_ort").Path.Replace("\\", "/")
    if ($cachedHome -ne $expectedHome) {
      Write-Host "CMake cache path changed, cleaning build dir: $buildDir"
      Remove-Item -Recurse -Force $buildDir
    }
  }
}

cmake -S sdk/rvc_sdk_ort -B $buildDir -G "Visual Studio 17 2022" -A x64 `
  -DONNXRUNTIME_ROOT="$ortRoot" `
  -DFAISS_ROOT="$faissRoot"

cmake --build $buildDir --config Release

# Stage runtime DLLs next to the demo exe for quick testing.
$demoDir = "$buildDir/Release"
if (Test-Path $demoDir) {
  Copy-Item -Force -Path "$ortRoot/lib/onnxruntime.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$ortRoot/lib/onnxruntime_providers_cuda.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  # CUDA Provider 依赖 onnxruntime_providers_shared.dll，否则 --cuda 会直接启用失败
  Copy-Item -Force -Path "$ortRoot/lib/onnxruntime_providers_shared.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/faiss.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/libblas.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/liblapack.dll" -Destination $demoDir -ErrorAction SilentlyContinue
  Write-Host "Staged runtime DLLs into: $demoDir"
}
