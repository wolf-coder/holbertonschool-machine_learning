import numpy as np

def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    
    try:
        # Using np.argmax to find the index of the maximum value in each column
        decoded_labels = np.argmax(one_hot, axis=0)
        return decoded_labels
    except Exception:
        return None
