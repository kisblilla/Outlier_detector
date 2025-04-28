import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from detector import OutlierDetector

# constants 
MIN_V = 0.0
MAX_V = 8449.3107

if __name__ == '__main__':
    YEAR = 2022
    SEQ_LEN = 5
    BURN_IN = SEQ_LEN
    THRESHOLD = 0.06
    WEIGHTS = 'models/autoencoder.weights.h5'

    # 1) load the pre-selected mouse series
    series = np.load("./test_data/selected_mouse.npy")

    # randomly select an end for this time series of tumor volumes.
    series = series[: random.randrange(BURN_IN, len(series))]

    # 2) inject an outlier into the selected time series
    inject_day = -1
    series[inject_day] = 1500.0


    # 3) build detector
    stats = {'min': MIN_V, 'max': MAX_V}
    det = OutlierDetector(
        weights_path=WEIGHTS,
        stats=stats,
        sequence_length=SEQ_LEN,
        threshold=THRESHOLD,
        burn_in=BURN_IN
    )

    # 4) detect
    start = time.perf_counter()
    vols_json  = json.dumps(series.tolist())
    flags_json = det.detect(vols_json)
    flags      = json.loads(flags_json)
    end = time.perf_counter()
    print(f"Detection block took {end - start:.6f} seconds")

    # 5) print day-by-day flags (booleans for detection)
    for day, (vol, is_out) in enumerate(zip(series, flags)):
        status = 'OUTLIER' if is_out else 'ok'
        note   = '<-- injected' if day == inject_day else ''
        print(f"Day {day:>3}: Volume={vol:8.2f} → {status} {note}")

    # 6) regenerate preds and errors for plotting
    norm = (series - MIN_V) / (MAX_V - MIN_V)
    window, preds, errors = [], [], []
    for i, v in enumerate(norm):
        window.append(v)
        if i < BURN_IN or len(window) < SEQ_LEN:
            preds.append(np.nan)
            errors.append(0.0)
        else:
            seq = np.array(window[-SEQ_LEN:])
            inp = seq.reshape(1, SEQ_LEN, 1)
            recon = det.model.predict(inp, verbose=0)
            p = recon[0, -1, 0]
            preds.append(p)
            errors.append(abs(seq[-1] - p))

    # 7) plot and save
    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(preds, label='Reconstruction')
    plt.plot(norm,  label='Normalized Original')
    plt.legend()
    plt.title('Realtime Reconstruction vs Original')
    plt.savefig('figures/realtime_recon.png')

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(errors)), errors)
    plt.axhline(THRESHOLD, linestyle='--')
    plt.title('Realtime Reconstruction Error')
    plt.savefig('figures/realtime_errors.png')
