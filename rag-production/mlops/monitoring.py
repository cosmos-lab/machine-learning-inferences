import time

def measure_latency(fn):
    start = time.time()
    result = fn()
    latency = time.time() - start
    return result, latency
