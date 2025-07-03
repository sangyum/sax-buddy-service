import pytest
import subprocess
import time
import socket
import os
from contextlib import contextmanager


def is_port_open(port: int) -> bool:
    """Check if a port is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0
    finally:
        sock.close()


@contextmanager
def firestore_emulator():
    """Context manager to start and stop Firestore emulator."""
    port = 8080
    
    # Check if emulator is already running
    if is_port_open(port):
        print(f"Firestore emulator already running on port {port}")
        yield
        return
    
    # Start emulator
    print(f"Starting Firestore emulator on port {port}")
    process = subprocess.Popen([
        'firebase', 'emulators:start', '--only', 'firestore', '--port', str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for emulator to start
    max_retries = 30
    for _ in range(max_retries):
        if is_port_open(port):
            print("Firestore emulator started successfully")
            break
        time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError("Failed to start Firestore emulator")
    
    try:
        yield
    finally:
        # Stop emulator
        print("Stopping Firestore emulator")
        process.terminate()
        process.wait()


@pytest.fixture(scope="session", autouse=True)
def setup_firestore_emulator():
    """Automatically start Firestore emulator for integration tests."""
    # Check if emulator is already running
    if is_port_open(8080):
        print("Firestore emulator already running")
        yield
        return
    
    # For CI/testing, assume emulator is managed externally
    # Users should start emulator manually: firebase emulators:start --only firestore
    yield


@pytest.fixture(autouse=True)
def setup_emulator_environment():
    """Set up environment variables for Firestore emulator."""
    os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "demo-test"
    yield
    # Clean up environment variables
    if "FIRESTORE_EMULATOR_HOST" in os.environ:
        del os.environ["FIRESTORE_EMULATOR_HOST"]
    if "GOOGLE_CLOUD_PROJECT" in os.environ:
        del os.environ["GOOGLE_CLOUD_PROJECT"]