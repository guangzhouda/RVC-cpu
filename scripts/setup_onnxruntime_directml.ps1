$ErrorActionPreference = "Stop"

# Download and extract:
# - Microsoft.ML.OnnxRuntime.DirectML (onnxruntime.dll with DML EP)
# - Microsoft.AI.DirectML (DirectML.dll redistributable)
#
# Notes:
# - This repo's C++ SDK is "runtime no Python". These downloads are runtime deps only.
# - DirectML is the recommended GPU path for older NVIDIA cards that cannot run ORT CUDA kernels (e.g. sm_50).

param(
  [string]$OrtDirectMLVersion = "1.17.1",
  [string]$DirectMLVersion = "1.15.4"
)

function Download-NuGetNupkg([string]$PackageIdLower, [string]$Version, [string]$OutFile) {
  $url = "https://api.nuget.org/v3-flatcontainer/$PackageIdLower/$Version/$PackageIdLower.$Version.nupkg"
  Write-Host "Downloading $PackageIdLower $Version"
  Write-Host "  $url"
  Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing
}

$downloads = "deps/_downloads"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

# 1) ORT DirectML
$ortId = "microsoft.ml.onnxruntime.directml"
$ortNupkg = Join-Path $downloads "onnxruntime-win-x64-directml-$OrtDirectMLVersion.nupkg"
Download-NuGetNupkg -PackageIdLower $ortId -Version $OrtDirectMLVersion -OutFile $ortNupkg

$ortDst = "deps/onnxruntime/onnxruntime-win-x64-directml-$OrtDirectMLVersion"
if (Test-Path $ortDst) { Remove-Item -Recurse -Force $ortDst }
New-Item -ItemType Directory -Force -Path $ortDst | Out-Null
Expand-Archive -Path $ortNupkg -DestinationPath $ortDst -Force
Write-Host "Extracted -> $ortDst"

# 2) DirectML redistributable
$dmlId = "microsoft.ai.directml"
$dmlNupkg = Join-Path $downloads "directml-$DirectMLVersion.nupkg"
Download-NuGetNupkg -PackageIdLower $dmlId -Version $DirectMLVersion -OutFile $dmlNupkg

$dmlDst = "deps/directml/directml-$DirectMLVersion"
if (Test-Path $dmlDst) { Remove-Item -Recurse -Force $dmlDst }
New-Item -ItemType Directory -Force -Path $dmlDst | Out-Null
Expand-Archive -Path $dmlNupkg -DestinationPath $dmlDst -Force
Write-Host "Extracted -> $dmlDst"

Write-Host "Done."

