from datetime import datetime


def ts():
    return datetime.now().timestamp()

def duration(start:float):
    return datetime.now().timestamp() - start