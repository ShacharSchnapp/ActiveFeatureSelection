from math import log

# Discrete estimators
def entropyd(sx, base=2):
    return entropyfromprobs(hist(sx), base=base)

def hist(sx):
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())


def entropyfromprobs(probs, base=2):
    return -sum(map(elog, probs))/log(base)


def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)