#!/bin/bash
# Crab Robot Unified Launcher
# Fixed: proper cleanup of local processes + host log streaming

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
DIM="\033[2m"
NC="\033[0m"
BOLD="\033[1m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIN_HOST="orin"
HOST_TIMEOUT=60
HOST_LOG="/tmp/crab_host.log"
LOCAL_HOST_LOG="/tmp/crab_host_remote.log"

TASK_NAMES=("pick_place" "bimanual_handover" "stack" "pour" "insertion" "free_manipulation")
HARDNESS_NAMES=("soft" "medium" "hard")

CHILD_PID=""
LOG_PID=""

# --- Activate Conda ---
source ~/miniconda3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to source conda.sh${NC}"
    exit 1
fi

conda activate mobile-robot
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate mobile-robot environment${NC}"
    exit 1
fi

start_host_on_orin() {
    echo -e "${CYAN}Starting host on Orin...${NC}"

    # Kill any existing host (synchronous - wait for it to actually die)
    ssh -o ConnectTimeout=5 $ORIN_HOST "pkill -9 -f crab_host; pkill -9 -f 'lerobot.robots.crab.crab_host'; sleep 1" 2>/dev/null
    sleep 1

    # Clear old log
    ssh -o ConnectTimeout=5 $ORIN_HOST "rm -f $HOST_LOG" 2>/dev/null

    # Start host in background
    ssh -n -o ConnectTimeout=10 $ORIN_HOST \
        "cd ~/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab && nohup bash start_host.sh > $HOST_LOG 2>&1 &" &
    sleep 3

    # Wait for host to be ready
    echo -n "  Waiting for host "
    local count=0
    while ! python "$SCRIPT_DIR/check_host.py" 2>/dev/null; do
        echo -n "."
        sleep 1
        count=$((count + 1))
        if [ $count -ge $HOST_TIMEOUT ]; then
            echo -e " ${RED}TIMEOUT${NC}"
            echo -e "${YELLOW}Host log (last 20 lines):${NC}"
            ssh -o ConnectTimeout=5 $ORIN_HOST "tail -20 $HOST_LOG" 2>/dev/null
            return 1
        fi
    done
    echo -e " ${GREEN}OK${NC}"

    # Start streaming host log in background
    echo -e "${DIM}  (Host log streaming to $LOCAL_HOST_LOG)${NC}"
    ssh -o ConnectTimeout=5 $ORIN_HOST "tail -f $HOST_LOG" > "$LOCAL_HOST_LOG" 2>/dev/null &
    LOG_PID=$!

    return 0
}

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # 1. Kill local Python child process
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "  Stopping local process (PID $CHILD_PID)..."
        kill -TERM "$CHILD_PID" 2>/dev/null
        # Give it 3s to clean up gracefully
        for i in 1 2 3; do
            kill -0 "$CHILD_PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still alive
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "  Force killing (PID $CHILD_PID)..."
            kill -9 "$CHILD_PID" 2>/dev/null
        fi
    fi

    # 2. Kill any stray teleoperate/record processes (belt and suspenders)
    pkill -9 -f "examples/crab/teleoperate.py" 2>/dev/null
    pkill -9 -f "examples/crab/record.py" 2>/dev/null

    # 3. Stop host log streaming
    if [ -n "$LOG_PID" ] && kill -0 "$LOG_PID" 2>/dev/null; then
        kill "$LOG_PID" 2>/dev/null
    fi

    # 4. Kill host on Orin (synchronous)
    echo "  Stopping host on Orin..."
    ssh -o ConnectTimeout=5 $ORIN_HOST "pkill -f crab_host; pkill -f 'lerobot.robots.crab.crab_host'" 2>/dev/null

    sleep 1
    echo -e "${GREEN}Cleanup complete.${NC}"
}
trap cleanup EXIT INT TERM

# --- Kill any leftover processes from previous runs ---
echo -e "${DIM}Cleaning up previous sessions...${NC}"
pkill -9 -f "examples/crab/teleoperate.py" 2>/dev/null
pkill -9 -f "examples/crab/record.py" 2>/dev/null
sleep 0.5

# --- UI ---
clear
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              CRAB ROBOT CONTROLLER                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}Select mode:${NC}"
echo "  1) Teleoperation"
echo "  2) Dataset Recording"
read -p "Choice [1-2]: " mode
[ "$mode" != "2" ] && mode="1"

if [ "$mode" = "2" ]; then
    echo ""
    echo -e "${BOLD}Select task:${NC}"
    for i in "${!TASK_NAMES[@]}"; do echo "  $i) ${TASK_NAMES[$i]}"; done
    read -p "Task [0-5]: " TASK_ID
    [[ ! "$TASK_ID" =~ ^[0-5]$ ]] && TASK_ID=5

    echo ""
    echo -e "${BOLD}Select hardness:${NC}"
    for i in "${!HARDNESS_NAMES[@]}"; do echo "  $i) ${HARDNESS_NAMES[$i]}"; done
    read -p "Hardness [0-2]: " HARDNESS_ID
    [[ ! "$HARDNESS_ID" =~ ^[0-2]$ ]] && HARDNESS_ID=1

    DEFAULT_NAME="${TASK_NAMES[$TASK_ID]}_${HARDNESS_NAMES[$HARDNESS_ID]}_$(date +%Y%m%d_%H%M%S)"
    read -p "Dataset name [$DEFAULT_NAME]: " DATASET_NAME
    [ -z "$DATASET_NAME" ] && DATASET_NAME=$DEFAULT_NAME

    read -p "Episodes [10]: " NUM_EPISODES
    [[ ! "$NUM_EPISODES" =~ ^[0-9]+$ ]] && NUM_EPISODES=10
fi

# --- Start Host ---
start_host_on_orin || exit 1

echo ""
echo -e "${DIM}Host log: tail -f $LOCAL_HOST_LOG${NC}"
echo ""

# --- Run Mode ---
if [ "$mode" = "2" ]; then
    echo -e "${GREEN}Starting dataset recording...${NC}"
    echo "  Task: [${TASK_ID}] ${TASK_NAMES[$TASK_ID]}"
    echo "  Hardness: [${HARDNESS_ID}] ${HARDNESS_NAMES[$HARDNESS_ID]}"
    echo "  Dataset: $DATASET_NAME"
    echo ""

    cd ~/AnywhereVLA/lerobot
    sudo -E env "PATH=$PATH" python examples/crab/record.py \
        --task-id "$TASK_ID" \
        --hardness-id "$HARDNESS_ID" \
        --num-episodes "$NUM_EPISODES" \
        --repo-id "$DATASET_NAME" &
    CHILD_PID=$!
    wait $CHILD_PID
else
    echo -e "${GREEN}Starting teleoperation...${NC}"
    echo "  Robot IP:     192.168.50.239"
    echo "  Left leader:  /dev/leader_left"
    echo "  Right leader: /dev/leader_right"
    echo ""

    cd ~/AnywhereVLA/lerobot
    python examples/crab/teleoperate.py &
    CHILD_PID=$!
    wait $CHILD_PID
fi
