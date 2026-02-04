$ErrorActionPreference = "Stop"

# Package the built artifacts into sdk/rvc_sdk_ort/dist/
# 说明：该包用于“给别的程序用”的最小交付，不包含模型文件。

function Package-One([string]$BuildDir, [string]$OutRoot, [string]$RuntimeName) {
  if (!(Test-Path $BuildDir)) {
    throw "Build output not found: $BuildDir (run scripts/build_rvc_sdk_ort.ps1 first)."
  }

  New-Item -ItemType Directory -Force -Path "$OutRoot/bin", "$OutRoot/lib", "$OutRoot/include" | Out-Null

  # Public headers
  Copy-Item -Force -Path "sdk/rvc_sdk_ort/include/rvc_sdk_ort.h" -Destination "$OutRoot/include/"

  # Import lib + DLL
  Copy-Item -Force -Path "build_rvc_sdk_ort/Release/rvc_sdk_ort.lib" -Destination "$OutRoot/lib/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/rvc_sdk_ort.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue

  # Demo binaries (optional)
  Copy-Item -Force -Path "$BuildDir/rvc_sdk_ort_demo.exe" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/rvc_sdk_ort_realtime.exe" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue

  # Runtime deps (ORT + FAISS)
  Copy-Item -Force -Path "$BuildDir/onnxruntime.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/DirectML.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue

  Copy-Item -Force -Path "$BuildDir/onnxruntime_providers_cuda.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/onnxruntime_providers_shared.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue

  Copy-Item -Force -Path "$BuildDir/faiss.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/libblas.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue
  Copy-Item -Force -Path "$BuildDir/liblapack.dll" -Destination "$OutRoot/bin/" -ErrorAction SilentlyContinue

  # Docs
  Copy-Item -Force -Path "sdk/rvc_sdk_ort/README.md" -Destination "$OutRoot/"
  Copy-Item -Force -Path "sdk/rvc_sdk_ort/API.md" -Destination "$OutRoot/" -ErrorAction SilentlyContinue

  Write-Host "Packaged ($RuntimeName) -> $OutRoot"
}

Package-One -BuildDir "build_rvc_sdk_ort/Release" -OutRoot "sdk/rvc_sdk_ort/dist/win-x64-cuda" -RuntimeName "cuda"

if (Test-Path "build_rvc_sdk_ort/Release_dml") {
  Package-One -BuildDir "build_rvc_sdk_ort/Release_dml" -OutRoot "sdk/rvc_sdk_ort/dist/win-x64-dml" -RuntimeName "dml"
}
