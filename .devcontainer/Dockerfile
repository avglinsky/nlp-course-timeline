FROM python:3.10

# Setup
RUN apt-get update && \
    apt-get -y install sudo && \
    useradd -m vscode -s /bin/bash && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

## Switch to the user
#USER vscode

# Set the working directory
WORKDIR /workspace

# Expose ports
#EXPOSE 3000

