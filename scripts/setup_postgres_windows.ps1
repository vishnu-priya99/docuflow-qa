<#
.SYNOPSIS
    Sets up a local PostgreSQL instance for this project on Windows WITHOUT
    Docker and WITHOUT admin rights - using EnterpriseDB's portable zip
    binary distribution (no installer, no Windows service).

    Idempotent: safe to re-run. Skips steps that are already done (existing
    extraction, existing data directory, server already running, roles/DB
    that already exist).

.PARAMETER InstallDir
    Where to extract the Postgres binaries + data directory. Defaults to
    a project-local, git-ignored folder so the whole app stays
    self-contained in this "new folder".

.PARAMETER Port
    Port for the server to listen on. Must match DATABASE_URL in .env.

.PARAMETER PgVersion
    EDB release to fetch, as it appears in their zip filenames
    (e.g. "17.6-1"). See https://www.enterprisedb.com/download-postgresql-binaries.

.PARAMETER Stop
    Stop the running server instead of setting anything up.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_postgres_windows.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_postgres_windows.ps1 -Stop
#>
param(
    [string]$InstallDir = (Join-Path $PSScriptRoot "..\.pgsql-portable"),
    [int]$Port = 5432,
    [string]$PgVersion = "17.6-1",
    [string]$SuperPassword = "postgres",
    [string]$AppPassword = "docqa",
    [string]$ReadonlyPassword = "docqa_ro",
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$BinDir = Join-Path $InstallDir "pgsql\bin"
$DataDir = Join-Path $InstallDir "data"
$LogFile = Join-Path $InstallDir "logfile.txt"
$PgCtl = Join-Path $BinDir "pg_ctl.exe"
$InitDb = Join-Path $BinDir "initdb.exe"
$Psql = Join-Path $BinDir "psql.exe"
$PgIsReady = Join-Path $BinDir "pg_isready.exe"

function Test-ServerRunning {
    if (-not (Test-Path $PgIsReady)) { return $false }
    & $PgIsReady -h localhost -p $Port *> $null
    return ($LASTEXITCODE -eq 0)
}

if ($Stop) {
    if (Test-Path $PgCtl) {
        & $PgCtl -D $DataDir stop
    } else {
        Write-Host "Nothing installed at $InstallDir"
    }
    return
}

# --- 1. Binaries ---
if (-not (Test-Path $BinDir)) {
    Write-Host "Downloading PostgreSQL $PgVersion portable binaries (~300MB)..."
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $zipPath = Join-Path $InstallDir "pg.zip"
    $url = "https://get.enterprisedb.com/postgresql/postgresql-$PgVersion-windows-x64-binaries.zip"
    # Prefer real curl.exe (bundled with Windows 10 1803+) - it's dramatically
    # faster here than Invoke-WebRequest, even with progress rendering off.
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail -o $zipPath $url
        if ($LASTEXITCODE -ne 0) { throw "curl.exe failed to download $url" }
    } else {
        $previousProgressPreference = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try {
            Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
        } finally {
            $ProgressPreference = $previousProgressPreference
        }
    }

    Write-Host "Extracting server binaries (skipping pgAdmin/StackBuilder GUI bundles)..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $needed = @("pgsql/bin/", "pgsql/lib/", "pgsql/share/")
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName.EndsWith("/")) { continue }
            if (-not ($needed | Where-Object { $entry.FullName.StartsWith($_) })) { continue }
            $destPath = Join-Path $InstallDir $entry.FullName
            $destDir = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
        }
    } finally {
        $zip.Dispose()
    }
    Remove-Item $zipPath -Force
    Write-Host "Binaries ready at $BinDir"
} else {
    Write-Host "Binaries already present at $BinDir - skipping download."
}

# --- 2. Data directory ---
if (-not (Test-Path $DataDir)) {
    Write-Host "Initializing data directory..."
    $pwFile = Join-Path $InstallDir "pwfile.tmp"
    Set-Content -Path $pwFile -Value $SuperPassword -NoNewline
    & $InitDb -D $DataDir -U postgres --pwfile=$pwFile -E UTF8
    Remove-Item $pwFile -Force
} else {
    Write-Host "Data directory already exists at $DataDir - skipping initdb."
}

# --- 3. Start server ---
if (Test-ServerRunning) {
    Write-Host "Server already running on port $Port."
} else {
    Write-Host "Starting server on port $Port..."
    & $PgCtl -D $DataDir -l $LogFile -o "-p $Port" start
    Start-Sleep -Seconds 2
    if (-not (Test-ServerRunning)) {
        throw "Server did not come up - check $LogFile"
    }
}

# --- 4. App role + database + read-only role ---
$env:PGPASSWORD = $SuperPassword
$roleExists = & $Psql -h localhost -p $Port -U postgres -d postgres -tA -c "SELECT 1 FROM pg_roles WHERE rolname='docqa'"
if ($roleExists -ne "1") {
    Write-Host "Creating role 'docqa'..."
    & $Psql -h localhost -p $Port -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE docqa LOGIN PASSWORD '$AppPassword' CREATEDB;"
}

$dbExists = & $Psql -h localhost -p $Port -U postgres -d postgres -tA -c "SELECT 1 FROM pg_database WHERE datname='docqa'"
if ($dbExists -ne "1") {
    Write-Host "Creating database 'docqa'..."
    & $Psql -h localhost -p $Port -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE docqa OWNER docqa;"
}

Write-Host "Applying read-only role (scripts/init_readonly_role.sql)..."
$readonlySql = Join-Path $PSScriptRoot "init_readonly_role.sql"
& $Psql -h localhost -p $Port -U postgres -d docqa -v ON_ERROR_STOP=1 -f $readonlySql | Out-Null
if ($ReadonlyPassword -ne "docqa_ro") {
    & $Psql -h localhost -p $Port -U postgres -d docqa -v ON_ERROR_STOP=1 -c "ALTER ROLE docqa_ro PASSWORD '$ReadonlyPassword';" | Out-Null
}
Remove-Item Env:\PGPASSWORD

Write-Host ""
Write-Host "Done. Put this in your .env:"
Write-Host "  DATABASE_URL=postgresql+asyncpg://docqa:$AppPassword@localhost:$Port/docqa"
Write-Host "  DATABASE_URL_READONLY=postgresql+asyncpg://docqa_ro:$ReadonlyPassword@localhost:$Port/docqa"
Write-Host ""
Write-Host "Stop the server later with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Stop"
