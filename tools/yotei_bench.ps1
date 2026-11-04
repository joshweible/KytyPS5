# Unattended Ghost of Yotei benchmark. The game boots straight into a repeatable scene, so this
# launches it, waits for the frame rate to settle after loading, measures a fixed number of
# KYTY_BENCH_STATS windows, appends one line to bench-results.csv and closes the game.
#
#   tools\yotei_bench.ps1 -Label base [-Runs 1] [-Override '--gpu','1'] [-Env @{K='V'}]
#
# KYTY_SHADER_STORE_KEY keeps the shader store across rebuilds. Change -StoreKey whenever the
# recompiler or the store's record format changes.
param(
	[Parameter(Mandatory)][string]$Label,
	[int]$Runs = 1,
	[string[]]$Override = @(),
	[hashtable]$Env = @{},
	[string]$Shot = '', # Saves the game window here at the end of the run.
	[int]$Windows = 4,
	[int]$WindowSeconds = 15,
	[int]$TimeoutSeconds = 420,
	[string]$StoreKey = '2',
	[string]$Build = '',
	[string]$Game = 'D:\dev\ps5\dumps\PPSA26344',
	[string]$Output = 'D:\dev\ps5\runs\bench-upstream'
)
$ErrorActionPreference = 'Stop'
if ($Build -eq '') { $Build = Join-Path $PSScriptRoot '..\_Build\windows' }
$exe = (Resolve-Path (Join-Path $Build 'kyty_emulator.exe')).Path

$arguments = @('--game', $Game, '--amd-cpu', '--shader-cache', '--readback-linear-images',
               '--skip-notice-screen', '--screen-width', '1920', '--screen-height', '1080') + $Override
$environment = @{ KYTY_SHADER_STORE_KEY = $StoreKey; KYTY_BENCH_STATS = "$WindowSeconds" }
foreach ($key in $Env.Keys) { $environment[$key] = $Env[$key] }
foreach ($key in $environment.Keys) { Set-Item "env:$key" $environment[$key] }

function Read-Windows([string]$log) {
	$result = New-Object System.Collections.ArrayList
	foreach ($line in @(Get-Content $log -ErrorAction SilentlyContinue)) {
		if ($line -match '^bench-stats: frames=(\d+) fps=([\d.]+) busy=([\d.]+) gap=([\d.]+) wait=([\d.]+) compile=([\d.]+) create=([\d.]+) faults=(\d+) submits=(\d+) draws=(\d+) dispatches=(\d+)') {
			[void]$result.Add([pscustomobject]@{ Frames = [int]$Matches[1]; Fps = [double]$Matches[2]
				Busy = [double]$Matches[3]; Gap = [double]$Matches[4]; Wait = [double]$Matches[5]
				Compile = [double]$Matches[6]; Create = [double]$Matches[7]; Faults = [double]$Matches[8]
				Submits = [double]$Matches[9]; Draws = [double]$Matches[10]; Dispatches = [double]$Matches[11] })
		}
	}
	return , $result
}

New-Item -ItemType Directory -Force $Output | Out-Null
$csv = Join-Path $Output 'bench-results.csv'
if (-not (Test-Path $csv)) {
	'time,label,run,commit,fps,min_fps,max_fps,frame_ms,gpu_busy_ms,gpu_gap_ms,gpu_wait_ms,compile_ms,create_ms,faults,submits,draws,dispatches,settle_s,status' | Set-Content $csv
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
	$settle_time = 0
	$status = 'timeout'
	while ($clock.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
		Start-Sleep 3
		if ($process.HasExited) { $status = 'exited 0x{0:X8}' -f $process.ExitCode; break }
		$all = Read-Windows $stdout
		if ($settled -lt 0) {
			# Menus draw little and compile stalls present nothing: settled is two steady windows of
			# real scene load.
			for ($i = 1; $i -lt $all.Count; $i++) {
				$a = $all[$i - 1]; $b = $all[$i]
				if ($a.Draws -ge 1000 -and $b.Draws -ge 1000 -and $a.Fps -gt 0 -and $b.Fps -gt 0 -and
				    [math]::Abs($a.Fps - $b.Fps) -le 0.15 * [math]::Max($a.Fps, $b.Fps)) { $settled = $i + 1; break }
			}
			if ($settled -ge 0) { $settle_time = [int]$clock.Elapsed.TotalSeconds }
		}
		if ($settled -ge 0 -and $all.Count -ge $settled + $Windows) { $status = 'ok'; break }
	}
	if ($Shot -and -not $process.HasExited) {
		Add-Type -AssemblyName System.Drawing
		Add-Type -Name Window -Namespace Bench -MemberDefinition @'
delegate bool EnumProc(IntPtr window, IntPtr parameter);
[DllImport("user32.dll")] static extern bool EnumWindows(EnumProc callback, IntPtr parameter);
[DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr window);
[DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr window, out uint id);
[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, int[] rect);
public static IntPtr Largest(uint process) {
	IntPtr best = IntPtr.Zero;
	long   area = 0;
	EnumWindows((window, parameter) => {
		uint id;
		GetWindowThreadProcessId(window, out id);
		var rect = new int[4];
		if (id == process && IsWindowVisible(window) && GetWindowRect(window, rect)) {
			long size = (long)(rect[2] - rect[0]) * (rect[3] - rect[1]);
			if (size > area) { area = size; best = window; }
		}
		return true;
	}, IntPtr.Zero);
	return best;
}
'@
		[void][Bench.Window]::SetProcessDPIAware() # Window rects are scaled otherwise.
		$handle = [Bench.Window]::Largest([uint32]$process.Id)
		[void][Bench.Window]::SetForegroundWindow($handle)
		Start-Sleep 2
		$rect = New-Object int[] 4
		[void][Bench.Window]::GetWindowRect($handle, $rect)
		$image = New-Object System.Drawing.Bitmap ($rect[2] - $rect[0]), ($rect[3] - $rect[1])
		$canvas = [System.Drawing.Graphics]::FromImage($image)
		$canvas.CopyFromScreen($rect[0], $rect[1], 0, 0, $image.Size)
		$image.Save($Shot)
	}
	if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force; $process.WaitForExit(30000) | Out-Null }
	$all = Read-Windows $stdout
	$line = "{0},{1},{2},{3}" -f (Get-Date -Format s), $Label, $run, $commit
	if ($status -eq 'ok') {
		$window = @($all[$settled..($settled + $Windows - 1)])
		$frames = ($window | Measure-Object Frames -Sum).Sum
		$avg = { param($n) (($window | ForEach-Object { $_.$n * $_.Frames } | Measure-Object -Sum).Sum) / $frames }
		$fps = $frames / ($Windows * $WindowSeconds)
		$range = $window | ForEach-Object Fps | Measure-Object -Minimum -Maximum
		$line += ",{0:F2},{1:F1},{2:F1},{3:F1},{4:F1},{5:F1},{6:F1},{7:F1},{8:F1},{9:F0},{10:F0},{11:F0},{12:F0},{13},ok" -f `
		    $fps, $range.Minimum, $range.Maximum, (1000 / $fps), (& $avg 'Busy'), (& $avg 'Gap'), (& $avg 'Wait'),
		    (& $avg 'Compile'), (& $avg 'Create'), (& $avg 'Faults'), (& $avg 'Submits'), (& $avg 'Draws'),
		    (& $avg 'Dispatches'), $settle_time
	} else {
		$line += ",,,,,,,,,,,,,,$status"
	}
	$line | Add-Content $csv
	Write-Output $line
}
