# Exit immediately if any command fails
set -e

# Change directory to the mounted home partition containing the project codebase
cd /home/hlcv_team013

# Export the active Weights & Biases workspace entity name to the environment
export WANDB_ENTITY="default-entity" # Placeholder

# Include user-level local binary path to resolve location warnings
export PATH="/home/hlcv_team013/.local/bin:$PATH"

# Append the project root folder to the Python path to resolve data module imports
export PYTHONPATH="/home/hlcv_team013:$PYTHONPATH"

echo "=== System: Installing dependencies from requirements.txt ==="
/opt/conda/bin/pip install --no-cache-dir -r requirements.txt

echo "=== System: Adjusting library versions to align with PyTorch 2.3.1 ==="
# Pinning peft down alongside transformers prevents the EncoderDecoderCache import error
/opt/conda/bin/pip install --no-cache-dir "transformers<4.43.0" "accelerate<0.33.0" "peft<0.12.0" datasets

echo "=== System: Executing main pipeline pipeline... ==="
# Forward all received HTCondor submission arguments to the runtime interpreter
/opt/conda/bin/python "$@"