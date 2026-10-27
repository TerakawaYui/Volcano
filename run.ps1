param(
    [Parameter(Mandatory=$true)][string]$FitsPath,
    [string]$OutputDir,
    [ValidateRange(1,32)][int]$Workers=3,
    [Nullable[double]]$BeamArcsec,
    [string]$Config,
    [string]$BeamMap,
    [switch]$NoFigures
)
$ErrorActionPreference='Stop'
$pythonPath=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements.txt first; see README.md.' }
$source=(Resolve-Path -LiteralPath $FitsPath -ErrorAction Stop).Path
if(-not $OutputDir) {
    $stamp=[DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('yyyyMMdd_HHmmss')
    $name=[IO.Path]::GetFileNameWithoutExtension($source.TrimEnd('\'))
    $OutputDir=Join-Path $PSScriptRoot "outputs\${name}_$stamp"
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir=(Resolve-Path -LiteralPath $OutputDir).Path
$argsList=@('-u',(Join-Path $PSScriptRoot 'main.py'),$source,'--output',$OutputDir,'--workers',"$Workers")
if($null -ne $BeamArcsec) { $argsList+=@('--beam-arcsec',"$BeamArcsec") }
if($Config) { $argsList+=@('--config',(Resolve-Path -LiteralPath $Config).Path) }
if($BeamMap) { $argsList+=@('--beam-map',(Resolve-Path -LiteralPath $BeamMap).Path) }
elseif((Get-Item -LiteralPath $source).PSIsContainer) { $argsList+=@('--beam-map',(Join-Path $PSScriptRoot 'settings\archive_beams.json')) }
if($NoFigures) { $argsList+='--no-figures' }
$env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONIOENCODING='utf-8'
Write-Host "Output: $OutputDir"
Write-Host 'To resume, repeat the same command with -OutputDir set to this directory.'
$ErrorActionPreference='Continue'
& $pythonPath @argsList 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append
$runExit=$LASTEXITCODE
if($runExit -ne 0) { Write-Error "Search failed ($runExit); inspect run.log and manifest.json." }
exit $runExit
