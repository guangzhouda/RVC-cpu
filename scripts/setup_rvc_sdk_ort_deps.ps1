$ErrorActionPreference = "Stop"

# This script prepares local dependencies for sdk/rvc_sdk_ort:
# - onnxruntime (Windows x64 GPU zip, includes CPU EP + CUDA EP)
# - libfaiss (CPU build) via micromamba, then copied into deps/faiss
#
# It does NOT install Python, Visual Studio, or CUDA drivers/toolkit.
# Python is only used here to extract a .tar.bz2.

$repoRoot = (Get-Location).Path

New-Item -ItemType Directory -Force -Path "deps", "deps/_downloads", "deps/onnxruntime", "deps/micromamba", "deps/faiss" | Out-Null

function Download-File {
  param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$OutFile
  )
  if (Test-Path $OutFile) {
    Write-Host "Skip download (exists): $OutFile"
    return
  }
  Write-Host "Downloading: $Url"
  & curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 2 -o $OutFile $Url
}

function Extract-TarBz2-WithPython {
  param(
    [Parameter(Mandatory=$true)][string]$TarBz2,
    [Parameter(Mandatory=$true)][string]$DestDir
  )
  New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
  @"
import tarfile
from pathlib import Path
pkg = Path(r"$TarBz2")
dst = Path(r"$DestDir")
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(pkg, "r:bz2") as tf:
    tf.extractall(dst)
print("extracted", pkg, "->", dst)
"@ | python -
}

# 1) onnxruntime
$ortVer = "1.17.1"
$ortZip = "deps/_downloads/onnxruntime-win-x64-gpu-$ortVer.zip"
$ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$ortVer/onnxruntime-win-x64-gpu-$ortVer.zip"
Download-File -Url $ortUrl -OutFile $ortZip

Write-Host "Extracting ORT zip -> deps/onnxruntime ..."
& tar -xf $ortZip -C "deps/onnxruntime"

$ortRoot = "deps/onnxruntime/onnxruntime-win-x64-gpu-$ortVer"
if (!(Test-Path "$ortRoot/include/onnxruntime_cxx_api.h")) {
  throw "ORT not extracted correctly: missing $ortRoot/include/onnxruntime_cxx_api.h"
}

# 2) micromamba (from conda-forge)
$mmVer = "2.5.0-1"
$mmPkg = "deps/_downloads/micromamba-$mmVer.tar.bz2"
$mmUrl = "https://conda.anaconda.org/conda-forge/win-64/micromamba-$mmVer.tar.bz2"
Download-File -Url $mmUrl -OutFile $mmPkg

Write-Host "Extracting micromamba -> deps/micromamba (via Python tarfile) ..."
Extract-TarBz2-WithPython -TarBz2 $mmPkg -DestDir "deps/micromamba"

$mmExe = "deps/micromamba/Library/bin/micromamba.exe"
if (!(Test-Path $mmExe)) {
  throw "micromamba.exe not found at: $mmExe"
}
& $mmExe --version

# 3) libfaiss (CPU build) via micromamba into a local prefix, then copy into deps/faiss
New-Item -ItemType Directory -Force -Path "deps/micromamba/root" | Out-Null
$env:MAMBA_ROOT_PREFIX = (Resolve-Path "deps/micromamba/root").Path

$faissPrefix = "deps/micromamba/env-faiss-1.9.0-cpu"
Write-Host "Installing libfaiss=1.9.0 (cpu) -> $faissPrefix ..."
& $mmExe create -y -p $faissPrefix -c conda-forge "libfaiss=1.9.0=*_cpu"

New-Item -ItemType Directory -Force -Path "deps/faiss/include", "deps/faiss/lib", "deps/faiss/bin" | Out-Null
Copy-Item -Recurse -Force -Path "$faissPrefix/Library/include/faiss" -Destination "deps/faiss/include"
Copy-Item -Force -Path "$faissPrefix/Library/lib/faiss.lib" -Destination "deps/faiss/lib"
Copy-Item -Force -Path "$faissPrefix/Library/bin/faiss.dll" -Destination "deps/faiss/bin"
# faiss.dll 依赖 BLAS/LAPACK（用于某些索引类型/距离计算），一起拷贝到运行目录，避免 demo/DLL 启动失败。
Copy-Item -Force -Path "$faissPrefix/Library/bin/libblas.dll" -Destination "deps/faiss/bin" -ErrorAction SilentlyContinue
Copy-Item -Force -Path "$faissPrefix/Library/bin/liblapack.dll" -Destination "deps/faiss/bin" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Done."
Write-Host "ONNXRUNTIME_ROOT = $ortRoot"
Write-Host "FAISS_ROOT       = deps/faiss"
