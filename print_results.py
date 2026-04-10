import numpy as np, os

raw = os.path.join('results', 'raw', 'engineering')
problems = {
    1: 'Spring (3D)',
    2: 'PressureVessel (4D)',
    3: 'WeldedBeam (4D)',
    4: 'SpeedReducer (7D)',
    5: 'Bearing (10D)',
}

print(f"{'F#':<4} {'Problem':<22} {'Algo':<6} {'Mean':>14} {'Std':>12} {'Best':>14} {'Worst':>14}")
print('-' * 86)
for fid in range(1, 6):
    for algo in ['COA', 'MCOA']:
        path = os.path.join(raw, f'{algo}_F{fid}_DEng.npy')
        d    = np.load(path)
        fits = d[:, 0]
        print(f"F{fid:<3} {problems[fid]:<22} {algo:<6} {fits.mean():>14.4e} {fits.std():>12.4e} {fits.min():>14.4e} {fits.max():>14.4e}")
    print()
