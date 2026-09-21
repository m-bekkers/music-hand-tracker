#!/bin/bash

set -e

TEMP_ROOT="/mnt/c/Users/matth/AppData/Local/Temp"
STATE_FILE="$TEMP_ROOT/webcam_server_state"

SOURCE="$1"

# ------------------------------------------------------------
# TEARDOWN
# ------------------------------------------------------------

if [ "$SOURCE" = "--teardown" ]; then

    if [ ! -f "$STATE_FILE" ]; then
        echo "No running instance found."
        exit 0
    fi

    echo "Reading state..."
    source "$STATE_FILE"

    # --------------------------------------------------------
    # Stop Windows process
    # --------------------------------------------------------

    if [ -n "$WINDOWS_PID" ]; then
        echo "Stopping Windows Python process (PID $WINDOWS_PID)..."

        taskkill.exe //PID "$WINDOWS_PID" //T //F 2>/dev/null || true

        # Wait for the process to actually disappear
        for i in {1..20}; do
            if ! powershell.exe -NoProfile -Command \
                "Get-Process -Id $WINDOWS_PID -ErrorAction SilentlyContinue" \
                | grep -q .; then
                break
            fi

            sleep 0.5
        done
    fi

    # --------------------------------------------------------
    # Remove Windows directory
    # --------------------------------------------------------

    echo "Removing temporary directory..."

    powershell.exe -NoProfile -Command \
        "Remove-Item -LiteralPath '$WINDOWS_PROJECT_DIR' -Recurse -Force -ErrorAction Stop"

    # --------------------------------------------------------
    # Remove state
    # --------------------------------------------------------

    rm -f "$STATE_FILE"

    echo "Teardown complete."

    exit 0
fi


# ------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------

if [ -z "$SOURCE" ]; then
    echo "Usage:"
    echo "  $0 <python_file>"
    echo "  $0 --teardown"
    exit 1
fi

if [ ! -f "$SOURCE" ]; then
    echo "Error: File '$SOURCE' does not exist."
    exit 1
fi


# ------------------------------------------------------------
# LOCATE REQUIREMENTS
# ------------------------------------------------------------

SCRIPT_DIR="$(dirname "$(realpath "$SOURCE")")"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi


# ------------------------------------------------------------
# CREATE TEMPORARY PROJECT
# ------------------------------------------------------------

FILENAME="$(basename "$SOURCE")"

PROJECT_DIR="$(mktemp -d "$TEMP_ROOT/webcam_server_XXXXXX")"

echo "Created temporary directory:"
echo "  $PROJECT_DIR"


# ------------------------------------------------------------
# COPY FILES
# ------------------------------------------------------------

cp "$SOURCE" "$PROJECT_DIR/$FILENAME"
cp "$SCRIPT_DIR/requirements.txt" "$PROJECT_DIR/requirements.txt"


# ------------------------------------------------------------
# WINDOWS PATHS
# ------------------------------------------------------------

WINDOWS_PROJECT_DIR="$(wslpath -w "$PROJECT_DIR")"
WINDOWS_SCRIPT="$(wslpath -w "$PROJECT_DIR/$FILENAME")"
WINDOWS_REQUIREMENTS="$(wslpath -w "$PROJECT_DIR/requirements.txt")"


# ------------------------------------------------------------
# CREATE WINDOWS VENV
# ------------------------------------------------------------

echo
echo "Creating Windows virtual environment..."

python.exe -m venv "$WINDOWS_PROJECT_DIR/.venv"


# ------------------------------------------------------------
# INSTALL DEPENDENCIES
# ------------------------------------------------------------

echo
echo "Installing dependencies..."

"$PROJECT_DIR/.venv/Scripts/python.exe" \
    -m pip install -r "$WINDOWS_REQUIREMENTS"


# ------------------------------------------------------------
# START WINDOWS PYTHON
# ------------------------------------------------------------

echo
echo "Starting Windows Python..."

nohup "$PROJECT_DIR/.venv/Scripts/python.exe" \
    "$WINDOWS_SCRIPT" \
    > "$PROJECT_DIR/server.log" 2>&1 &

WSL_PID=$!

# Give Windows a moment to start the process
sleep 1


# ------------------------------------------------------------
# FIND WINDOWS PID
# ------------------------------------------------------------

WINDOWS_PID="$(
    powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.ProcessId -ne 0 -and \$_.CommandLine -like '*$FILENAME*' } | Select-Object -First 1 -ExpandProperty ProcessId)"
)"

WINDOWS_PID="$(echo "$WINDOWS_PID" | tr -d '\r')"


# ------------------------------------------------------------
# STORE TEARDOWN STATE
# ------------------------------------------------------------

cat > "$STATE_FILE" <<EOF
PROJECT_DIR="$PROJECT_DIR"
WINDOWS_PROJECT_DIR="$WINDOWS_PROJECT_DIR"
WINDOWS_SCRIPT="$WINDOWS_SCRIPT"
WINDOWS_PID="$WINDOWS_PID"
WSL_PID="$WSL_PID"
FILENAME="$FILENAME"
EOF

echo
echo "Windows Python process started."
echo "  Windows PID: $WINDOWS_PID"
echo "  WSL PID:     $WSL_PID"
echo "  Project:     $PROJECT_DIR"
echo "  State:       $STATE_FILE"
echo "  Log:         $PROJECT_DIR/server.log"

echo
echo "run.sh exiting."

exit 0