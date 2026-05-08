from datetime import datetime


def ts():
    return datetime.now().timestamp()

def duration(start:float):
    return datetime.now().timestamp() - start




def get_trainset(filename: str, count=10 ** 9) -> list[str]:
    """ Reads the file line after line"""
    res = []
    with open(filename, "r") as f:
        for ln in f:
            if not ln.strip(): continue
            res.append(ln.strip())
            if len(res) >= count:
                break
    return res
