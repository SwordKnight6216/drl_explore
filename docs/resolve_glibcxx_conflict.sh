#!/bin/bash

# Initial Problem: You were trying to run a Python script in a Conda environment, but encountered a libGL error related to a mismatch in GLIBCXX versions.
# Initial Analysis: We identified that the Conda environment's libstdc++.so.6 library was conflicting with the system's version.
# Solution Steps:
# Updated the system's libstdc++6.
# To ensure the system's version was used over the Conda version, the libstdc++.so.6 file in the Conda environment was renamed.
# After these changes, the script executed successfully.


# Update system's libstdc++6
echo "Updating system's libstdc++6..."
sudo apt-get update
sudo apt-get upgrade -y libstdc++6

# Rename the libstdc++.so.6 in the Conda environment to force use of the system's version
CONDA_ENV_PATH="/home/swordknight/miniconda3/envs/drl"
if [ -f "$CONDA_ENV_PATH/lib/libstdc++.so.6" ]; then
    echo "Renaming libstdc++.so.6 in the Conda environment..."
    mv "$CONDA_ENV_PATH/lib/libstdc++.so.6" "$CONDA_ENV_PATH/lib/libstdc++.so.6.bak"
fi

echo "All done! You can now run your Python script without the GLIBCXX version conflict."

# Grant execute permissions: chmod +x resolve_glibcxx_conflict.sh.
# Run it: ./resolve_glibcxx_conflict.sh.
