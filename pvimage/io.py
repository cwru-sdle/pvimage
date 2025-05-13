import numpy as np
from skimage.color import gray2rgb

def read_txt_file(file, rgb=False):
    rows = []
    max_len = 0

    # Read and parse each line
    with open(file, 'r', encoding='latin1', errors='ignore') as f:
        for line in f:
            try:
                values = [float(x) for x in line.strip().split()]
                rows.append(values)
                max_len = max(max_len, len(values))
            except ValueError:
                continue  # Skip malformed lines

    # Pad rows with zeros
    for i in range(len(rows)):
        if len(rows[i]) < max_len:
            rows[i].extend([0.0] * (max_len - len(rows[i])))

    img = np.array(rows)

    if rgb:
        return gray2rgb(img)
    else:
        return img
