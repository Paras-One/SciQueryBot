# Use the official Python image as a base image
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN guardrails hub install hub://guardrails/toxic_language 

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
