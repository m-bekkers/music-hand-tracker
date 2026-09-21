$out = usbipd list

# Find the Connected devices header
$header = $out | Where-Object {
    $_ -match '^\s*BUSID\s+VID:PID\s+DEVICE\s+STATE'
}

if (-not $header) {
    Write-Error "Could not find usbipd device header."
    exit 1
}

# Find the starting position of each column
$busid_start  = $header.IndexOf("BUSID")
$vidpid_start = $header.IndexOf("VID:PID")
$device_start = $header.IndexOf("DEVICE")
$state_start  = $header.IndexOf("STATE")

# Output BUSID and DEVICE for every connected USB device
foreach ($line in $out) {

    # USB device rows start with a BUSID such as 3-11, 3-12, 6-1, etc.
    if ($line -match '^\s*\d+-\d+') {

        $busid = $line.Substring(
            $busid_start,
            $vidpid_start - $busid_start
        ).Trim()

        $device = $line.Substring(
            $device_start,
            $state_start - $device_start
        ).Trim()

        # Output in a format that is easy for Bash to parse
        Write-Output "$busid|$device"
    }
}