# hermes/metrics.py
from prometheus_client import start_http_server, Summary
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

def process_request():
    with REQUEST_TIME.time():
        time.sleep(1)  # Simulate processing

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        process_request()

