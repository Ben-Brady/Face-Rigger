import time

class Timer:
    def __init__(self,delay:float):
        self.start = time.time()
        self.delay = delay / 1000

    def __call__(self):
        duration = time.time() - self.start
        delay = self.delay - duration
        if delay > 0:
            time.sleep(self.delay)
            
        self.start = time.time()
        return True