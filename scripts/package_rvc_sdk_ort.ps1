$ErrorActionPreference = "Stop"

# Package the built artifacts into sdk/rvc_sdk_ort/dist/win-x64
# 说明：该包用于“给别的程序用”的最小交付，不包含模型文件。

$outRoot = "sdk/rvc_sdk_ort/dist/win-x64"
$buildDir = "build_rvc_sdk_ort/Release"

if (!(Test-Path $buildDir)) {
  throw "Build output not found: $buildDir (run scripts/build_rvc_sdk_ort.ps1 first)."
}

New-Item -ItemType Directory -Force -Path "$outRoot/bin", "$outRoot/lib", "$outRoot/include" | Out-Null

# Public headers
Copy-Item -Force -Path "sdk/rvc_sdk_ort/include/rvc_sdk_ort.h" -Destination "$outRoot/include/"

# Import lib + DLL
Copy-Item -Force -Path "$buildDir/rvc_sdk_ort.lib" -Destination "$outRoot/lib/"
Copy-Item -Force -Path "$buildDir/rvc_sdk_ort.dll" -Destination "$outRoot/bin/"

# Demo binaries (optional)
Copy-Item -Force -Path "$buildDir/rvc_sdk_ort_demo.exe" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$buildDir/rvc_sdk_ort_realtime.exe" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue

# Runtime deps (ORT + FAISS)
Copy-Item -Force -Path "$buildDir/onnxruntime.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$buildDir/onnxruntime_providers_cuda.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$buildDir/onnxruntime_providers_shared.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue

Copy-Item -Force -Path "$buildDir/faiss.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$buildDir/libblas.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$buildDir/liblapack.dll" -Destination "$outRoot/bin/" -ErrorAction SilentlyContinue

# Docs
Copy-Item -Force -Path "sdk/rvc_sdk_ort/README.md" -Destination "$outRoot/"

Write-Host "Packaged -> $outRoot"
