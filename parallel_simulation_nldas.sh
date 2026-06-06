#!/bin/bash

# ── User configuration: edit these paths for your system ───────────────────
# HRLDAS run directory holding hrldas.exe, namelist.hrldas, and NoahmpTable.TBL
WORK_DIR="${WORK_DIR:-/path/to/hrldas/run}"
# Root for per-member output; one sim<N>/ subdirectory is created per member
ENSEMBLE_OUTPUT_DIR="${ENSEMBLE_OUTPUT_DIR:-/path/to/ensemble}"
# Root of the per-member atmospheric forcing inputs
FORCING_BASE_DIR="${FORCING_BASE_DIR:-/path/to/forcing}"

# Number of simulations to run concurrently per batch
MAX_PARALLEL_JOBS=32

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
    
    # Map this simulation to its forcing member (members are odd-numbered: 1, 3, 5, ...)
    forcing_sim_num=$((2 * sim_num - 1))

    # Create the new namelist, keeping the original content except what we need to change
    awk -v setup_file="${FORCING_BASE_DIR}/forcing_${start_date}_sim${forcing_sim_num}/HRLDAS_setup_${setup_date}_d1" \
        -v indir="${FORCING_BASE_DIR}/forcing_${start_date}_sim${forcing_sim_num}/" \
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
        print " SPINUP_LOOPS = 3"
        print " KDAY = 14"
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
current_date="2022-03-21_00"
sim_count=0

echo "Starting batch processing with $MAX_PARALLEL_JOBS jobs per batch"

# Process in batches
while [ $(date -d "${current_date/_/ }" +%s) -lt $(date -d "2022-09-07 21:00:00" +%s) ]; do
    batch_start=$((sim_count + 1))
    echo "Starting batch beginning with simulation $batch_start"
    
    # Start a batch of jobs
    batch_count=0
    batch_date=$current_date
    
    while [ $batch_count -lt $MAX_PARALLEL_JOBS ] && 
          [ $(date -d "${batch_date/_/ }" +%s) -lt $(date -d "2022-09-21 21:00:00" +%s) ]; do
        ((sim_count++))
        ((batch_count++))
        
        echo "Starting job $sim_count (batch job $batch_count) for date $batch_date"
        run_simulation "$batch_date" "$sim_count" &
        
        # Advance 6 hours to the next ensemble start time
        batch_date=$(date -d "${batch_date/_/ } + 6 hour" "+%Y-%m-%d_%H")
    done
    
    # Store the current date for next batch
    current_date=$batch_date
    
    echo "Waiting for batch (simulations $batch_start to $sim_count) to complete..."
    wait
    echo "Batch completed. Moving to next batch..."
    echo "----------------------------------------"
done

echo "All $sim_count simulations completed!"
