#!/bin/bash

# Test mode - set to false for full run
TEST_MODE=true
TEST_HOURS=5

# ── User configuration: edit these paths for your system ───────────────────
# HRLDAS run directory holding hrldas.exe, namelist.hrldas, and NoahmpTable.TBL
WORK_DIR="${WORK_DIR:-/path/to/hrldas/run}"
# Root for per-member output; one sim<N>/ subdirectory is created per member
ENSEMBLE_OUTPUT_DIR="${ENSEMBLE_OUTPUT_DIR:-/path/to/ensemble}"
# Root of the per-member atmospheric forcing inputs
FORCING_BASE_DIR="${FORCING_BASE_DIR:-/path/to/forcing}"

# Number of simulations to run concurrently (kept low for testing)
MAX_PARALLEL_JOBS=3

# Function to format date for setup filename (YYYYMMDDHH)
format_date_for_setup() {
    local date_str=$1
    local year=${date_str:0:4}
    local month=${date_str:5:2}
    local day=${date_str:8:2}
    local hour=${date_str:11:2}
    echo "${year}${month}${day}${hour}"
}

# Function to run a single simulation
run_simulation() {
    local start_date=$1
    local sim_num=$2
    
    echo "Starting simulation $sim_num for date $start_date"
    
    # Create work directory
    local work_dir="${WORK_DIR}/work_${sim_num}"
    mkdir -p "$work_dir"
    cd "$work_dir"
    
    # Create output directory
    local output_dir="${ENSEMBLE_OUTPUT_DIR}/sim${sim_num}/"
    mkdir -p "$output_dir"
    
    # Parse date components
    local year=${start_date:0:4}
    local month=${start_date:5:2}
    local day=${start_date:8:2}
    local hour=${start_date:11:2}
    
    # Format date for setup file
    local setup_date=$(format_date_for_setup "$start_date")
    
    # Copy executable and namelist, create symlink for NoahmpTable.TBL
    cp ../hrldas.exe .
    ln -sf ../NoahmpTable.TBL .
    
    # Create the new namelist, keeping the original content except what we need to change
    awk -v setup_file="${FORCING_BASE_DIR}/forcing_${start_date}_sim${sim_num}/HRLDAS_setup_${setup_date}_d1" \
        -v indir="${FORCING_BASE_DIR}/forcing_${start_date}_sim${sim_num}/" \
        -v outdir="${output_dir}" \
        -v year="$year" \
        -v month="$month" \
        -v day="$day" \
        -v hour="$hour" '
    BEGIN {in_offline_section=0}
    /&NOAHLSM_OFFLINE/ {
        print $0
        in_offline_section=1
        print " HRLDAS_SETUP_FILE = \"" setup_file "\""
        print " INDIR = \"" indir "\""
        print " OUTDIR = \"" outdir "\""
        print " START_YEAR  = " year
        print " START_MONTH = " month
        print " START_DAY   = " day
        print " START_HOUR  = " hour
        print " START_MIN   = 00"
        print " SPINUP_LOOPS = 2"
        print " KDAY = 3"
        next
    }
    in_offline_section && /^[ ]*START_/ {next}
    in_offline_section && /^[ ]*HRLDAS_SETUP_FILE/ {next}
    in_offline_section && /^[ ]*INDIR/ {next}
    in_offline_section && /^[ ]*OUTDIR/ {next}
    in_offline_section && /^[ ]*SPINUP_LOOPS/ {next}
    in_offline_section && /^[ ]*KDAY/ {next}
    {print}' ../namelist.hrldas > namelist.hrldas
    
    # Save a copy of the namelist to the output directory
    cp namelist.hrldas "${output_dir}/namelist.hrldas.${sim_num}"
    
    echo "Created namelist for simulation $sim_num (saved to ${output_dir}/namelist.hrldas.${sim_num})"
    cat namelist.hrldas
    echo "----------------------------------------"
    
    # Run the simulation
    ./hrldas.exe > "${output_dir}/log_${sim_num}.txt" 2>&1
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "Successfully completed simulation $sim_num"
        cd ..
        rm -rf "$work_dir"
        return 0
    else
        echo "ERROR: Simulation $sim_num failed! Check ${output_dir}/log_${sim_num}.txt"
        cd ..
        return 1
    fi
}

# Generate dates and run simulations
current_date="2024-07-01_00"
sim_count=0

echo "Test mode: Will run $TEST_HOURS simulations"
echo "First few simulations will be:"

while [ $sim_count -lt $TEST_HOURS ]; do
    ((sim_count++))
    echo "Sim $sim_count: $current_date"
    echo "  Setup file: HRLDAS_setup_$(format_date_for_setup "$current_date")_d1"
    echo "  Output dir: ${ENSEMBLE_OUTPUT_DIR}/sim${sim_count}/"
    current_date=$(date -d "${current_date/_/ } + 1 hour" "+%Y-%m-%d_%H")
done

# Ask for confirmation
echo -n "Do these paths look correct? (y/n) "
read answer
if [ "$answer" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

# Reset for actual run
current_date="2024-07-01_00"
sim_count=0

# Run simulations
while [ $sim_count -lt $TEST_HOURS ]; do
    ((sim_count++))
    
    echo "Adding job $sim_count: $current_date"
    
    # Run simulation in background
    run_simulation "$current_date" "$sim_count" &
    
    # Limit number of parallel jobs
    if [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; then
        wait -n
    fi
    
    # Move to next hour
    current_date=$(date -d "${current_date/_/ } + 1 hour" "+%Y-%m-%d_%H")
done

# Wait for remaining jobs to complete
wait

echo "All $sim_count test simulations completed!"
echo "Check the output directories and logs to verify results"
