# Unattended Ghost of Yotei benchmark. The game boots straight into the campfire scene, so this
# launches it, waits for the frame rate to settle after loading, measures a fixed number of
# --drain-stats windows, appends one line to bench-results.csv and closes the game.
#
#   tools\yotei_bench.ps1 -Label base [-Runs 3] [-Override '--relaxed-readback','true'] [-Env @{K='V'}]
param(
	[Parameter(Mandatory)][string]$Label,
	[int]$Runs = 1,
	[string[]]$Override = @(),
	[hashtable]$Env = @{},
	[int]$Windows = 4,
	[int]$WindowSeconds = 15,
	[int]$TimeoutSeconds = 420,
	[string]$Build = '',
	[string]$Game = 'D:\dev\ps5\dumps\PPSA26344',
	[string]$Output = 'D:\dev\ps5\runs\bench'
)
$ErrorActionPreference = 'Stop'
if ($Build -eq '') { $Build = Join-Path $PSScriptRoot '..\_Build\windows' }
$exe =(Resolve-Path (Join-Path $Build 'kyty_emulator.exe')).Path

$settings = [ordered]@{
	'--readback-linear-images' = 'true'; '--screen-width' = '1920'; '--screen-height' = '1080'
	'--game-preset-priority' = 'Performance'; '--gpu-frames-ahead' = '0'
	'--gpu-timestamp-scale' = '200'; '--label-flush-interval-us' = '2000'
	'--async-pipelines' = 'false'; '--defer-dispatches' = 'false'; '--relaxed-readback' = 'false'
	'--speculative-draws' = 'true'; '--record-thread' = 'true'
}
for ($i = 0; $i + 1 -lt $Override.Count; $i += 2) { $settings[$Override[$i]] = $Override[$i + 1] }
$arguments = @('--game', $Game, '--amd-cpu', '--drain-stats', "$WindowSeconds")
foreach ($key in $settings.Keys) { $arguments += $key; $arguments += $settings[$key] }

$environment = @{
	KYTY_COMPUTE_CHAIN = '1'; KYTY_READBACK_QUEUE = '0'; KYTY_READBACK_TARGETED_PUBLICATION = '1'
	KYTY_GPU_PALETTE_SANITIZE = '0'; KYTY_BINDLESS_KEY_BOUNDS = '1'; KYTY_BINDLESS_EAGER = '1'
	KYTY_BINDLESS_TSHARP_CHECK = '1'; KYTY_AVPLAYER_RECYCLE = '1'; KYTY_PRESENT_WHEN_READY = '1'
}
foreach ($key in $Env.Keys) { $environment[$key] = $Env[$key] }
foreach ($key in $environment.Keys) { Set-Item "env:$key" $environment[$key] }

function Read-Windows([string]$log) {
	# One object per finished drain-stats window; its detail rows follow the header line.
	$lines = @(Get-Content $log -ErrorAction SilentlyContinue)
	$result = New-Object System.Collections.ArrayList
	$current = $null
	foreach ($line in $lines) {
		if ($line -match '^drain-stats: [\d.]+s frames=(\d+) \(([\d.]+)/s\).*full-drain n=\d+ ([\d.]+)ms .*tick-wait n=\d+ ([\d.]+)ms') {
			$current = [pscustomobject]@{ Frames = [int]$Matches[1]; Fps = [double]$Matches[2]
				FullDrain = [double]$Matches[3]; TickWait = [double]$Matches[4]
				Busy = 0.0; Gap = 0.0; Blocked = 0.0; Compile = 0.0 }
			[void]$result.Add($current)
		} elseif ($current -and $line -match '^  (gpu-busy|gpu-gap|blocked-poll|shader-compile|pipeline-create)\s.*?\s([\d.]+)ms') {
			$value = [double]$Matches[2]
			switch ($Matches[1]) {
				'gpu-busy' { $current.Busy += $value }
				'gpu-gap' { $current.Gap += $value }
				'blocked-poll' { $current.Blocked += $value }
				default { $current.Compile += $value }
			}
		}
	}
	return , $result
}

New-Item -ItemType Directory -Force $Output | Out-Null
$csv = Join-Path $Output 'bench-results.csv'
if (-not (Test-Path $csv)) {
	'time,label,run,commit,fps,min_fps,max_fps,frame_ms,gpu_busy_ms,gpu_gap_ms,blocked_ms,full_drain_ms,tick_wait_ms,compile_ms,settle_s,status' | Set-Content $csv
}
$commit = (git -C (Split-Path $exe) rev-parse --short HEAD 2>$null)

for ($run = 1; $run -le $Runs; $run++) {
	# A crashed or killed instance holds the guest memory reservation until it has fully exited.
	for ($wait = 0; $wait -lt 60 -and (Get-Process kyty_emulator -ErrorAction SilentlyContinue); $wait++) { Start-Sleep 1 }
	if (Get-Process kyty_emulator -ErrorAction SilentlyContinue) { throw 'kyty_emulator is still running' }
	$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
	$stdout = Join-Path $Output "$Label-$run-$stamp.log"
	$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory (Split-Path $exe) `
	    -RedirectStandardOutput $stdout -RedirectStandardError "$stdout.err" -PassThru
	$null = $process.Handle
	$clock = [Diagnostics.Stopwatch]::StartNew()
	$settled = -1
	$status = 'timeout'
	while ($clock.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
		Start-Sleep 3
		if ($process.HasExited) { $status = 'exited 0x{0:X8}' -f $process.ExitCode; break }
		$all = Read-Windows $stdout
		if ($settled -lt 0) {
			# Menus run fast and compile stalls present nothing: settled is two steady, slow windows.
			for ($i = 1; $i -lt $all.Count; $i++) {
				$a = $all[$i - 1]; $b = $all[$i]
				if ($a.Fps -gt 0 -and $b.Fps -gt 0 -and $a.Fps -lt 15 -and $b.Fps -lt 15 -and
				    [math]::Abs($a.Fps - $b.Fps) -le 0.25 * [math]::Max($a.Fps, $b.Fps)) { $settled = $i + 1; break }
			}
			if ($settled -ge 0) { $settle_time = [int]$clock.Elapsed.TotalSeconds }
		}
		if ($settled -ge 0 -and $all.Count -ge $settled + $Windows) { $status = 'ok'; break }
	}
	if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force; $process.WaitForExit(30000) | Out-Null }
	$all = Read-Windows $stdout
	$line = "{0},{1},{2},{3}" -f (Get-Date -Format s), $Label, $run, $commit
	if ($status -eq 'ok') {
		$window = @($all[$settled..($settled + $Windows - 1)])
		$frames = ($window | Measure-Object Frames -Sum).Sum
		$sum = { param($n) ($window | Measure-Object $n -Sum).Sum / $frames }
		$fps = $frames / ($Windows * $WindowSeconds)
		$fpsList = $window | ForEach-Object Fps | Measure-Object -Minimum -Maximum
		$line += ",{0:F2},{1:F1},{2:F1},{3:F1},{4:F1},{5:F1},{6:F1},{7:F1},{8:F1},{9:F1},{10},ok" -f `
		    $fps, $fpsList.Minimum, $fpsList.Maximum, (1000 / $fps), (& $sum 'Busy'), (& $sum 'Gap'),
		    (& $sum 'Blocked'), (& $sum 'FullDrain'), (& $sum 'TickWait'), (& $sum 'Compile'), $settle_time
	} else {
		$line += ",,,,,,,,,,,,$status"
	}
	$line | Add-Content $csv
	Write-Output $line
}
