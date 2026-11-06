#!/usr/bin/env python
"""Write a small synthetic test case for AR-Decon (64^3, 1.5 A/pixel):
half1.mrc, half2.mrc, full.mrc and a soft spherical mask.mrc in OUTDIR.
A few Gaussian 'atoms' blurred anisotropically (worse along z), plus
independent noise in the two half-maps.  Used by the CI smoke test."""
import sys, os
import numpy as np

def write_mrc(path, vol, apix):
    """Minimal MRC2014 writer (mode 2, little endian)."""
    vol = np.ascontiguousarray(vol, dtype='<f4'); nz, ny, nx = vol.shape
    hdr = bytearray(1024)
    hdr[0:40] = np.array([nx, ny, nz, 2, 0, 0, 0, nx, ny, nz], '<i4').tobytes()
    hdr[40:64] = np.array([nx * apix, ny * apix, nz * apix, 90, 90, 90], '<f4').tobytes()
    hdr[64:76] = np.array([1, 2, 3], '<i4').tobytes()
    hdr[76:88] = np.array([vol.min(), vol.max(), vol.mean()], '<f4').tobytes()
    hdr[208:216] = b'MAP DA\x00\x00'
    hdr[216:220] = np.array([vol.std()], '<f4').tobytes()
    open(path, 'wb').write(bytes(hdr) + vol.tobytes())

def main(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    n, apix = 64, 1.5
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(float)
    vol = np.zeros((n, n, n))
    for _ in range(60):
        c = rng.uniform(20, 44, 3)
        vol += np.exp(-((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2) / (2 * 1.2**2))
    k = np.fft.fftfreq(n)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    blur = np.exp(-(kx**2 + ky**2) / (2 * 0.12**2) - kz**2 / (2 * 0.06**2))
    vol = np.fft.ifftn(np.fft.fftn(vol) * blur).real
    h1 = vol + 0.05 * rng.standard_normal(vol.shape)
    h2 = vol + 0.05 * rng.standard_normal(vol.shape)
    r = np.sqrt((x - 32)**2 + (y - 32)**2 + (z - 32)**2)
    mask = np.clip((26 - r) / 4, 0, 1)
    for name, v in [('half1', h1), ('half2', h2), ('full', (h1 + h2) / 2), ('mask', mask)]:
        write_mrc(os.path.join(out, name + '.mrc'), v, apix)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'small')
