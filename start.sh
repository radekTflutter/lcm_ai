#!/bin/bash
# Startup script for Adaptive Sentinel AI Factory

echo "Starting Adaptive Sentinel AI Factory..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "Running migrations..."
python manage.py makemigrations
python manage.py migrate

# Create media directories
mkdir -p media/datasets
mkdir -p media/models
mkdir -p media/backgrounds

# Check CUDA availability
echo "Checking CUDA..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Start server
echo "Starting server..."
echo "Access the application at: http://localhost:8000"
python manage.py runserver 0.0.0.0:8000
