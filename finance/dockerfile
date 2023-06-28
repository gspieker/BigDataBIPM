# Build stage 1: Base image
FROM python:3.6 as base

# Set the working directory inside the container
WORKDIR /finance

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Build stage 2: Final image
FROM base as final

# Copy the rest of the application code to the working directory
COPY . /app
WORKDIR /app

# Expose port 8501 to the outside world
EXPOSE 8501

# Set the command to run when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
