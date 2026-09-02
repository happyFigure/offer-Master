param(
    [switch]$PrintEnvironment
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VendorRoot = Join-Path $ProjectRoot "vendor\external-skills\xiaohongshu-mcp-complete"
$BinaryPath = Join-Path $VendorRoot "bin\xiaohongshu-login-windows-amd64.exe"
$RuntimeRoot = Join-Path $ProjectRoot ".runtime\xiaohongshu-mcp"

$RuntimeEnv = [ordered]@{
    COOKIES_PATH    = Join-Path $RuntimeRoot "data\cookies.json"
    LOCALAPPDATA    = Join-Path $RuntimeRoot "localappdata"
    APPDATA         = Join-Path $RuntimeRoot "appdata"
    TEMP            = Join-Path $RuntimeRoot "temp"
    TMP             = Join-Path $RuntimeRoot "temp"
    HOME            = Join-Path $RuntimeRoot "home"
    XDG_CACHE_HOME  = Join-Path $RuntimeRoot "xdg-cache"
    XDG_CONFIG_HOME = Join-Path $RuntimeRoot "xdg-config"
}

foreach ($path in $RuntimeEnv.Values) {
    $dir = if ([System.IO.Path]::GetExtension($path)) { Split-Path -Parent $path } else { $path }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

foreach ($entry in $RuntimeEnv.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
}

if ($PrintEnvironment) {
    [ordered]@{
        runtime_root = $RuntimeRoot
        binary_path  = $BinaryPath
        env          = $RuntimeEnv
    } | ConvertTo-Json -Depth 5
    exit 0
}

if (-not (Test-Path $BinaryPath)) {
    throw "xiaohongshu login binary not found: $BinaryPath"
}

Write-Host "Starting xiaohongshu login. Runtime data: $RuntimeRoot"
& $BinaryPath
