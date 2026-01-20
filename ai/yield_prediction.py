import numpy as np


def predict_yield(temp, humidity, soil):
    # Simple formula for demo
    yield_estimate = (temp + humidity + soil) / 3
    return round(yield_estimate, 2)
