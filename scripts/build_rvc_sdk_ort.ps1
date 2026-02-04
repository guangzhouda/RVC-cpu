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

# 额外：DirectML 运行时（用于老卡/非 CUDA 环境）
# 说明：CUDA 版与 DML 版的 onnxruntime.dll 导出集合不同，不能放在同一目录下“二选一”。
# 这里把 DML 运行时整理到 Release_dml，方便直接运行：
#   build_rvc_sdk_ort/Release_dml/rvc_sdk_ort_realtime.exe --dml ...
$dmlOrtRoot = "deps/onnxruntime/onnxruntime-win-x64-directml-1.17.1"
$dmlOrtDll = Join-Path $dmlOrtRoot "runtimes/win-x64/native/onnxruntime.dll"
$dmlDir = "$buildDir/Release_dml"
if ((Test-Path $demoDir) -and (Test-Path $dmlOrtDll)) {
  New-Item -ItemType Directory -Force -Path $dmlDir | Out-Null
  Copy-Item -Force -Path "$demoDir/rvc_sdk_ort.dll" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$demoDir/rvc_sdk_ort_realtime.exe" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$demoDir/rvc_sdk_ort_demo.exe" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$demoDir/rvc_sdk_ort_file.exe" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/faiss.dll" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/libblas.dll" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$faissRoot/bin/liblapack.dll" -Destination $dmlDir -ErrorAction SilentlyContinue
  Copy-Item -Force -Path $dmlOrtDll -Destination $dmlDir -ErrorAction SilentlyContinue

  # 可选：把 DirectML redistributable 带上，避免某些系统 DirectML.dll 版本过老导致 DML EP 初始化失败。
  $directmlDll = Get-ChildItem -Path "deps/directml" -Recurse -Filter "DirectML.dll" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*\bin\x64-win\DirectML.dll" } |
    Select-Object -First 1
  if ($directmlDll) {
    Copy-Item -Force -Path $directmlDll.FullName -Destination $dmlDir -ErrorAction SilentlyContinue
  }

  Write-Host "Staged DirectML runtime into: $dmlDir"
}
