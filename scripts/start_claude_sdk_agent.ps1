param(
    [string]$AgentRoot = "",
    [string]$PythonExe = "E:\zte\lastest_bizcore\my-agents\.venv\Scripts\python.exe",
    [string]$ClaudeCliPath = "E:\zte\lastest_bizcore\my-agents\.venv\Lib\site-packages\claude_agent_sdk\_bundled\claude.exe",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 18008,
    [string]$TempRoot = "E:\zte\tmp",
    [string]$PipCacheRoot = "E:\zte\pip-cache"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$vendoredAgentRoot = Join-Path $repoRoot "vendor\claude-sdk-agent"
if (-not $AgentRoot) {
    $externalAgentRoot = "E:\zte\lastest_bizcore\claude-sdk-agent"
    if (Test-Path -LiteralPath (Join-Path $externalAgentRoot "src\main.py")) {
        $AgentRoot = $externalAgentRoot
    }
    else {
        $AgentRoot = $vendoredAgentRoot
    }
}

$resolvedAgentRoot = Resolve-Path -LiteralPath $AgentRoot
if (-not (Test-Path -LiteralPath (Join-Path $resolvedAgentRoot "src\main.py"))) {
    throw "claude-sdk-agent source not found at $resolvedAgentRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $ClaudeCliPath)) {
    throw "Native Claude CLI executable not found: $ClaudeCliPath"
}

New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PipCacheRoot | Out-Null

$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:PIP_CACHE_DIR = $PipCacheRoot
$env:PIP_NO_CACHE_DIR = "1"
$env:CLAUDE_SDK_AGENT_CLI_PATH = $ClaudeCliPath

$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1
if ($existing) {
    Write-Host "claude-sdk-agent already appears to be listening on port $Port (pid $($existing.OwningProcess))."
    exit 0
}

$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("src/main.py", "serve", "--host", $HostName, "--port", [string]$Port) `
    -WorkingDirectory $resolvedAgentRoot `
    -RedirectStandardOutput (Join-Path $logDir "claude-sdk-agent.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "claude-sdk-agent.stderr.log") `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started claude-sdk-agent pid=$($process.Id) url=http://$HostName`:$Port"
