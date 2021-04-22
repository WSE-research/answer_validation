import math

def NDCG(values, total_right):
    """
    calculates Normalized Discounted Cumulative Gain @ K
    values - contains max of K values in range [0..1], each represents answer rating 
    total_right - number of right answers in whole set
    """
    assert(all(i >= 0 and i <= 1 for i in values))
    ideal = sum(1/math.log(i+2) for i in range(min(len(values), total_right)))
    real = sum((2**x-1)/math.log(i+2) for i, x in enumerate(values[:total_right]))
    return real/ideal if ideal else 0

def NDCG_binary(values, total_right):
    """
    calculates Normalized Discounted Cumulative Gain @ K
    values - contains max of K values in set {0, 1}, each represents answer rating 
    total_right - number of right answers in whole set
    """
    assert(all(i == 0 or i == 1 for i in values))
    ideal = sum(1/math.log(i+2) for i in range(min(len(values), total_right)))
    real = sum(x/math.log(i+2) for i, x in enumerate(values[:total_right]))
    return real/ideal if ideal else 0

assert (NDCG([1, 1, 0], 2) == 1)
assert (NDCG_binary([1, 1, 0], 2) == 1)
