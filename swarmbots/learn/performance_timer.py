import time
from typing import Self


class PerformanceTimer:

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
    
    def start(self) -> Self:
        self.start_time = time.perf_counter()
        self.end_time = None
        return self
    
    def stop(self) -> Self:
        assert self.start_time is not None, 'Timer has not been started'
        self.end_time = time.perf_counter()
        return self
    
    def get_duration(self) -> float:
        assert self.end_time is not None, 'Timer has not been stopped'
        return self.end_time - self.start_time
    
    def __enter__(self) -> Self:
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
    
    def __call__(self) -> float:
        return self.get_duration()