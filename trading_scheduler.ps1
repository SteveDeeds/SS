# Trading Automation Scheduler
# Runs different trading scripts based on the day of the week
# Monday-Friday: Mobile signals and server
# Saturday: Optimization scripts in parallel
# Sunday: Optimal strategy finder

param(
    [switch]$Test,  # Test mode - shows what would run without executing
    [switch]$Force, # Force execution regardless of day
    [string]$Day    # Override day for testing (Monday, Tuesday, etc.)
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "python"
$LogFile = Join-Path $ScriptDir "automation_log.txt"

# Function to write log messages
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry
}

# Function to run Python script with error handling
function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string]$Description,
        [switch]$Background
    )
    
    $FullPath = Join-Path $ScriptDir $ScriptPath
    if (-not (Test-Path $FullPath)) {
        Write-Log "Script not found: $FullPath" "ERROR"
        return $false
    }
    
    Write-Log "Starting: $Description"
    
    if ($Test) {
        Write-Log "TEST MODE: Would run: $PythonExe $FullPath" "TEST"
        return $true
    }
    
    try {
        if ($Background) {
            # Run in background and return the job
            $Job = Start-Job -ScriptBlock {
                param($Python, $Script, $WorkDir)
                Set-Location $WorkDir
                & $Python $Script
            } -ArgumentList $PythonExe, $FullPath, $ScriptDir -Name $Description
            
            Write-Log "Started background job: $Description (Job ID: $($Job.Id))"
            return $Job
        }
        else {
            # Run synchronously
            Push-Location $ScriptDir
            $Result = & $PythonExe $FullPath
            Pop-Location
            
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Completed successfully: $Description"
                return $true
            }
            else {
                Write-Log "Failed with exit code ${LASTEXITCODE}: $Description" "ERROR"
                return $false
            }
        }
    }
    catch {
        Write-Log "Exception running $Description`: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Function to run optimization scripts in parallel
function Start-OptimizationSuite {
    Write-Log "Starting Saturday optimization suite..."
    
    $OptimizationJobs = @()
    
    # Start all optimization scripts in parallel
    $Scripts = @(
        @{Path="examples\bollinger_bands_optimization.py"; Name="Bollinger Bands Optimization"},
        @{Path="examples\ma_optimization.py"; Name="MA Optimization"},
        @{Path="examples\rsi_optimization.py"; Name="RSI Optimization"}
    )
    
    foreach ($Script in $Scripts) {
        $Job = Invoke-PythonScript -ScriptPath $Script.Path -Description $Script.Name -Background
        if ($Job) {
            $OptimizationJobs += $Job
        }
    }
    
    if (-not $Test -and $OptimizationJobs.Count -gt 0) {
        Write-Log "Waiting for optimization jobs to complete..."
        
        # Wait for all jobs to complete
        $OptimizationJobs | Wait-Job | Out-Null
        
        # Check results and clean up
        foreach ($Job in $OptimizationJobs) {
            $JobResult = Receive-Job $Job
            if ($Job.State -eq "Completed") {
                Write-Log "Optimization job completed: $($Job.Name)"
            }
            else {
                Write-Log "Optimization job failed: $($Job.Name) - State: $($Job.State)" "ERROR"
            }
            Remove-Job $Job
        }
        
        Write-Log "All optimization jobs completed"
    }
}

# Function to handle weekday tasks (Monday-Friday)
function Start-WeekdayTasks {
    Write-Log "Starting weekday tasks..."
    
    # Always run mobile_signals.py first
    $SignalsResult = Invoke-PythonScript -ScriptPath "mobile_signals.py" -Description "Mobile Signals Generation"
    
    if (-not $SignalsResult) {
        Write-Log "Mobile signals generation failed, skipping server start" "ERROR"
        return
    }
    
    # Always try to start mobile server - if it's already running, it will fail gracefully
    Write-Log "Starting mobile server (will skip if already running)..."
    if (-not $Test) {
        # Start server in background - don't worry about return value
        $ServerJob = Invoke-PythonScript -ScriptPath "mobile_server.py" -Description "Mobile Server" -Background
        if ($ServerJob) {
            Write-Log "Mobile server start attempted"
        }
    }
}

# Function to handle Sunday tasks
function Start-SundayTasks {
    Write-Log "Starting Sunday tasks..."
    Invoke-PythonScript -ScriptPath "examples\optimal_strategy_finder.py" -Description "Optimal Strategy Finder"
}

# Main execution logic
function Main {
    Write-Log "=== Trading Automation Scheduler Started ==="
    Write-Log "Script Directory: $ScriptDir"
    Write-Log "Test Mode: $Test"
    Write-Log "Force Mode: $Force"
    
    # Determine what day to use
    $CurrentDay = if ($Day) { $Day } else { (Get-Date).DayOfWeek.ToString() }
    Write-Log "Current Day: $CurrentDay"
    
    # Execute based on day of week
    switch ($CurrentDay) {
        {$_ -in @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")} {
            Write-Log "Weekday detected: $CurrentDay"
            Start-WeekdayTasks
        }
        
        "Saturday" {
            Write-Log "Saturday detected: Running optimization suite"
            Start-OptimizationSuite
        }
        
        "Sunday" {
            Write-Log "Sunday detected: Running optimal strategy finder"
            Start-SundayTasks
        }
        
        default {
            Write-Log "Unknown day: $CurrentDay" "ERROR"
            exit 1
        }
    }
    
    Write-Log "=== Trading Automation Scheduler Completed ==="
}

# Run main function
try {
    Main
}
catch {
    Write-Log "Fatal error: $($_.Exception.Message)" "ERROR"
    exit 1
}
