"""Keep normalized progress scores distinct from completed task predicates."""
import math
import numpy as np


def within_time_limit(elapsed, limit):
    """Allow one nanosecond for accumulated clock rounding, never a physics step."""
    return bool(math.isfinite(elapsed) and math.isfinite(limit) and elapsed <= limit+1e-9)


def task_result(value):
    if isinstance(value,(bool,np.bool_)):
        return {'source_check':bool(value),'source_check_kind':'predicate','source_success':bool(value)}
    score=float(value)
    if not math.isfinite(score):raise ValueError('Task check returned a non-finite score')
    return {'source_check':score,'source_check_kind':'score','source_success':bool(score>=1.-1e-9)}
