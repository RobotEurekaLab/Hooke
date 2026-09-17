"""Synthesised score + SFX + voiceover mix for the teaser."""
import math
import os
import wave
import numpy as np
from scipy import signal
from scipy.io import wavfile
from timeline import SCENES, TOTAL

SR = 44100
N = int(TOTAL * SR)
rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x, axis=-1)


def add(buf, x, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= buf.shape[1]:
        return
    x = x[..., :buf.shape[1] - i]
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    if x.ndim == 1:
        buf[0, i:i + x.shape[-1]] += x * gain * l * 1.414
        buf[1, i:i + x.shape[-1]] += x * gain * r * 1.414
    else:
        buf[:, i:i + x.shape[-1]] += x * gain


def env(n, a=0.01, d=0.1, s=0.7, r=0.05):
    t = np.arange(n) / SR
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip((dur - t) / max(r, 1e-4), 0, 1)
    return (e * rel).astype(np.float32)


def saw(freq, n, phase0=None, vib=0.0, vib_rate=5.5):
    t = np.arange(n) / SR
    f = freq * (1 + vib * np.sin(2 * math.pi * vib_rate * t + rng.random() * 6))
    ph = np.cumsum(f) / SR + (rng.random() if phase0 is None else phase0)
    return (2 * (ph % 1.0) - 1).astype(np.float32)


def section(freq, dur, voices=5, detune=0.012, vib=0.004):
    n = int(dur * SR)
    x = np.zeros(n, np.float32)
    for v in range(voices):
        x += saw(freq * (1 + detune * (v - (voices - 1) / 2) / voices), n, vib=vib, vib_rate=5 + rng.random())
    return x / voices


def reverb(x, secs=2.2, wet=0.3, pre=0.02):
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for c in range(2):
        ir = rng.normal(0, 1, n) * np.exp(-t * 6.9 / secs)
        ir = filt(ir, "lowpass", 6000)
        ir[: int(pre * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        out[c] = signal.fftconvolve(x[c], ir, mode="full")[: x.shape[1]]
    return x * (1 - wet) + out * wet * 2.2


# ------------------------------------------------------------------ part 1: orchestral
def part1():
    buf = np.zeros((2, N), np.float32)
    bpm = 100
    e8 = 60 / bpm / 2
    chords = [(50, [50, 53, 57]), (46, [46, 50, 53]), (43, [43, 46, 50]), (45, [45, 49, 52])]  # Dm Bb Gm A
    bar = 60 / bpm * 4
    end = 9.9
    # low drone
    dr = section(mtof(38), end, 6, 0.01) + 0.6 * section(mtof(45), end, 6, 0.01)
    dr = filt(dr, "lowpass", 700)
    sw = np.linspace(0.25, 1.0, dr.size) ** 1.5
    add(buf, dr * sw * env(dr.size, 3.0, 1, 1, 0.3), 0.0, 0.7)
    # cello ostinato
    pattern = [0, 0, 7, 0, 12, 0, 7, 3]
    t = 0.5
    i = 0
    while t < end - 0.05:
        bi = int((t - 0.5) // bar) % 4
        root, triad = chords[bi]
        iv = pattern[i % 8]
        note = root + (iv if iv != 3 else (triad[1] - root))
        x = section(mtof(note), e8 * 0.95, 4, 0.01, 0.0)
        x = filt(x, "lowpass", 900 + 1200 * t / end)
        acc = 1.0 if i % 8 in (0, 4) else 0.7
        vol = 0.14 + 0.1 * (t / end)
        add(buf, x * env(x.size, 0.012, 0.12, 0.45, 0.03) * acc, t, vol, pan=-0.3)
        t += e8
        i += 1
    # violins: tremolo chords from 10.5, spiccato 16ths from 20.5
    t = 4.1
    while t < end:
        bi = int((t - 0.5) // bar) % 4
        _, triad = chords[bi]
        seg = min(bar, end - t)
        for k, m in enumerate(triad):
            x = section(mtof(m + 24), seg, 5, 0.008, 0.006)
            trem = 0.6 + 0.4 * np.sin(2 * math.pi * 13 * np.arange(x.size) / SR) ** 2
            x = filt(x * trem, "bandpass", [500, 7000])
            g = 0.02 * (1 + 2.0 * max(0, (t - 4.1) / 6))
            add(buf, x * env(x.size, 0.4, 1, 1, 0.3), t, g, pan=0.25 + 0.2 * (k - 1))
        t += bar
    t = 6.1
    i = 0
    while t < end - 0.1:
        bi = int((t - 0.5) // bar) % 4
        _, triad = chords[bi]
        m = triad[i % 3] + 12 + (12 if i % 4 == 3 else 0)
        x = section(mtof(m), e8 / 2 * 0.8, 3, 0.006, 0)
        x = filt(x, "bandpass", [300, 6000])
        add(buf, x * env(x.size, 0.004, 0.04, 0.2, 0.02), t, 0.07 * (1 + max(0, t - 6.1) / 4), pan=0.4)
        t += e8 / 2
        i += 1
    # timpani on downbeats + hits at the cuts
    def timp(g):
        n = int(1.4 * SR)
        tt = np.arange(n) / SR
        f = 55 * (1 + 0.5 * np.exp(-tt * 30))
        x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-tt * 3.2)
        x += filt(rng.normal(0, 1, n), "lowpass", 400) * np.exp(-tt * 18) * 0.5
        return x.astype(np.float32) * g
    t = 4.1
    while t < end:
        add(buf, timp(0.35), t, 1.0)
        t += bar / 2
    for tc in (4.1, 6.1, 7.6):
        add(buf, timp(0.8), tc, 1.0)
        n = int(3 * SR)
        hit = filt(rng.normal(0, 1, n), "lowpass", 180) * np.exp(-np.arange(n) / SR * 2.5)
        add(buf, hit.astype(np.float32), tc, 0.35)
    # riser into the drop
    n = int(3.8 * SR)
    tt = np.arange(n) / SR
    noise = rng.normal(0, 1, n).astype(np.float32)
    rise = np.zeros(n, np.float32)
    for k in range(6):
        fc = 300 * 2 ** (tt / 3.8 * 4) * (1 + k * 0.5)
        rise += np.sin(2 * math.pi * np.cumsum(fc) / SR) * 0.08
    noise = filt(noise, "highpass", 1500) * (tt / 3.8) ** 3 * 0.25
    add(buf, (rise * (tt / 3.8) ** 2 + noise).astype(np.float32), 6.2, 0.8)
    return reverb(buf, 2.8, 0.35)


# ------------------------------------------------------------------ part 2: darksynth
def braam(t0):
    n = int(5.5 * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for m, g in ((26, 1.0), (33, 0.7), (38, 0.6), (45, 0.35)):
        for v in range(4):
            x += saw(mtof(m) * (1 + 0.006 * (v - 1.5)), n) * g
    cutoff_env = 120 + 2400 * np.exp(-tt * 1.2) * np.clip(tt * 20, 0, 1)
    # time-varying lowpass via chunks
    out = np.zeros(n, np.float32)
    ch = 2048
    zi = None
    for s in range(0, n, ch):
        fc = float(cutoff_env[s])
        b = sos("lowpass", min(fc, 18000), 4)
        seg = x[s:s + ch]
        if zi is None:
            zi = signal.sosfilt_zi(b) * 0
        y, zi = signal.sosfilt(b, seg, zi=zi)
        out[s:s + ch] = y
    out = np.tanh(out * 0.6) * np.exp(-tt * 0.55) * np.clip(tt * 30, 0, 1)
    sub = np.sin(2 * math.pi * 36.7 * tt) * np.exp(-tt * 0.8) * np.clip(tt * 40, 0, 1)
    return (out * 0.5 + sub * 0.6).astype(np.float32)


def glitch_sfx(dur=0.7, seed=0):
    g = np.random.default_rng(seed)
    n = int(dur * SR)
    x = np.zeros(n, np.float32)
    p = 0
    while p < n:
        L = int(g.integers(300, 3000))
        kind = g.integers(0, 3)
        tt = np.arange(L) / SR
        if kind == 0:
            seg = np.sign(np.sin(2 * math.pi * g.integers(200, 3000) * tt))
        elif kind == 1:
            seg = g.normal(0, 1, L)
        else:
            seg = np.round(np.sin(2 * math.pi * g.integers(60, 900) * tt) * 3) / 3
        reps = int(g.integers(1, 4))
        seg = np.tile(seg * (0.3 + 0.7 * g.random()), reps)[: n - p]
        x[p:p + seg.size] = seg
        p += seg.size + int(g.integers(0, 1500))
    return x * 0.5


def kick():
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    f = 48 + 170 * np.exp(-tt * 28)
    x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-tt * 7)
    x += filt(rng.normal(0, 1, n), "highpass", 3000) * np.exp(-tt * 200) * 0.4
    return np.tanh(x * 1.8).astype(np.float32)


def snare():
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    x = filt(rng.normal(0, 1, n), "bandpass", [900, 7000]) * np.exp(-tt * 14)
    x += np.sin(2 * math.pi * 190 * tt) * np.exp(-tt * 25) * 0.6
    return x.astype(np.float32)


def hat(open_=False):
    n = int((0.22 if open_ else 0.05) * SR)
    tt = np.arange(n) / SR
    return (filt(rng.normal(0, 1, n), "highpass", 7500) * np.exp(-tt * (14 if open_ else 90))).astype(np.float32)


def part2():
    buf = np.zeros((2, N), np.float32)
    drums = np.zeros((2, N), np.float32)
    t0 = 10.0
    beat = 0.5
    s16 = beat / 4
    end_beat = 34.4
    add(buf, braam(t0), t0, 0.9)
    add(buf, glitch_sfx(0.9, 1), t0 - 0.05, 0.35, pan=0.3)
    for tg, sd in ((17.0, 2), (22.6, 3), (27.6, 4)):
        add(buf, glitch_sfx(0.45, sd), tg - 0.15, 0.3, pan=-0.3)
    K, Sn, Hc, Ho = kick(), snare(), hat(), hat(True)
    kick_times = []
    # bars of 2 s from t0; drums enter one bar after the braam
    t = 11.0
    b = 0
    while t < end_beat:
        if not (28.15 <= t < 28.62):
            add(drums, K, t, 0.9)
            kick_times.append(t)
            if b % 2 == 1:
                add(drums, Sn, t, 0.45)
            for k in range(4):
                th = t + k * s16
                if k == 2:
                    add(drums, Ho, th, 0.08, pan=0.3)
                else:
                    add(drums, Hc, th, 0.06 + 0.03 * (k % 2), pan=0.3)
        t += beat
        b += 1
    # sidechain envelope from kicks
    sc = np.ones(N, np.float32)
    dk = int(0.3 * SR)
    shape = 1 - 0.75 * np.exp(-np.arange(dk) / SR * 14)
    for kt in kick_times:
        i = int(kt * SR)
        sc[i:i + dk] = np.minimum(sc[i:i + dk], shape[: max(0, min(dk, N - i))])
    # bass 16ths
    prog = [38, 38, 34, 36]  # D D Bb C
    t = 10.5
    i = 0
    while t < end_beat:
        if 28.15 <= t < 28.62:
            t += s16
            i += 1
            continue
        root = prog[int((t - t0) // 4) % 4]
        m = root + (12 if i % 4 == 2 else 0)
        n = int(s16 * 0.9 * SR)
        x = saw(mtof(m), n) + saw(mtof(m) * 1.005, n)
        x = filt(x, "lowpass", 300 + 900 * (1 if i % 4 == 0 else 0.4) * min(1, (t - 10.5) / 8 + 0.3))
        add(buf, np.tanh(x * 1.5) * env(n, 0.003, 0.06, 0.3, 0.01), t, 0.22)
        t += s16
        i += 1
    # pads
    chords = {38: [50, 53, 57, 62], 34: [46, 50, 53, 58], 36: [48, 52, 55, 60]}
    t = t0
    while t < 34.6:
        root = prog[int((t - t0) // 4) % 4]
        seg = 4.0
        x = np.zeros(int(seg * SR), np.float32)
        for m in chords[root]:
            x += section(mtof(m), seg, 3, 0.01, 0.003)
        x = filt(x, "lowpass", 1400)
        add(buf, x * env(x.size, 0.6, 1, 1, 0.6), t, 0.05, pan=0.0)
        t += seg
    # lead arp from 45.0
    arp = [62, 65, 69, 74, 69, 65, 62, 57]
    t = 17.0
    i = 0
    while t < end_beat:
        if 28.15 <= t < 28.62:
            t += s16
            i += 1
            continue
        root = prog[int((t - t0) // 4) % 4]
        m = arp[i % 8] + (root - 38)
        n = int(s16 * 0.8 * SR)
        x = saw(mtof(m), n) * 0.6 + np.sign(saw(mtof(m) * 0.5, n)) * 0.4
        x = filt(x, "lowpass", 1500 + 2500 * min(1, (t - 17) / 9))
        add(buf, x * env(n, 0.002, 0.05, 0.25, 0.01), t, 0.05, pan=0.35 * (1 if i % 2 else -1))
        t += s16
        i += 1
    buf *= sc[None, :]
    # impacts
    for ti, g in ((17.0, 0.5), (22.6, 0.6), (28.62, 1.0)):
        n = int(4 * SR)
        tt = np.arange(n) / SR
        boom = np.sin(2 * math.pi * np.cumsum(40 + 60 * np.exp(-tt * 8)) / SR) * np.exp(-tt * 1.4)
        crack = filt(rng.normal(0, 1, n), "bandpass", [200, 4000]) * np.exp(-tt * 9) * 0.5
        add(buf, (np.tanh((boom + crack) * 1.5)).astype(np.float32), ti, g * 0.6)
    # reverse swell into the title
    n = int(1.2 * SR)
    tt = np.arange(n) / SR
    sw = filt(rng.normal(0, 1, n), "bandpass", [400, 6000]) * (tt / 1.2) ** 4
    add(buf, sw.astype(np.float32), 27.45, 0.35)
    # end card drone
    n = int(6.5 * SR)
    x = np.zeros(n, np.float32)
    for m in (38, 45, 50, 57, 64):
        x += section(mtof(m), 6.5, 4, 0.008, 0.003)
    x = filt(x, "lowpass", 1100)
    add(buf, x * env(n, 0.8, 1, 1, 2.5), 33.1, 0.1)
    out = reverb(buf, 1.8, 0.22) + drums
    return out


# ------------------------------------------------------------------ voiceover
def load_vo(i, pitch=0.97):
    with wave.open(os.path.join(HERE, "build", f"vo{i}.wav")) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    nz = np.where(np.abs(a) > 0.01)[0]
    a = a[max(0, nz[0] - 200): nz[-1] + 2000]
    n_out = int(a.size / pitch * SR / sr)
    x = np.interp(np.linspace(0, a.size - 1, n_out), np.arange(a.size), a).astype(np.float32)
    x = filt(x, "highpass", 75)
    body = filt(x, "lowpass", 300)
    x = x + body * 0.8
    x = np.tanh(x * 3.2) / np.tanh(3.2)
    return x


def voiceover():
    buf = np.zeros((2, N), np.float32)
    ends = []
    for i, (name, start, dur, off) in enumerate(SCENES):
        if off is None:
            continue
        x = load_vo(i + 1)
        gain = 0.85 if start < 10.0 else 1.05
        if i == 7:  # robotic double, then the slow closing phrase
            tt = np.arange(x.size) / SR
            ring = x * np.sin(2 * math.pi * 55 * tt)
            add(buf, ring * 0.35, start + off + 0.012, 0.6 * gain, pan=-0.2)
            add(buf, x, start + off, gain)
            tail = load_vo(9, pitch=0.92)
            t_tail = start + off + x.size / SR + 0.3
            ttt = np.arange(tail.size) / SR
            add(buf, tail * np.sin(2 * math.pi * 52 * ttt) * 0.3, t_tail + 0.012, 0.6 * gain, pan=-0.2)
            add(buf, tail, t_tail, gain * 1.05)
            ends.append(("s8-tail", round(t_tail + tail.size / SR, 2), round(start + dur, 2)))
            continue
        add(buf, x, start + off, gain)
        ends.append((name, round(start + off + x.size / SR, 2), round(start + dur, 2)))
    wet = reverb(buf, 1.2, 0.12)
    return wet, ends


def main():
    vo, ends = voiceover()
    for e in ends:
        print("vo end vs scene end:", e)
    music = part1() + part2() * 0.85
    # duck music under the voice
    envv = np.abs(vo).mean(0)
    envv = signal.sosfilt(sos("lowpass", 4), envv)
    envv = np.clip(envv / (envv.max() * 0.25), 0, 1)
    duck = 1 - 0.55 * envv
    mix = music * duck[None, :] * 0.9 + vo * 1.5
    fade = np.ones(N, np.float32)
    fade[-int(1.5 * SR):] = np.linspace(1, 0, int(1.5 * SR))
    mix *= fade
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix /= np.abs(mix).max() / 0.95
    wavfile.write(os.path.join(HERE, "build", "mix.wav"), SR, (mix.T * 32767).astype(np.int16))
    print("wrote mix.wav", mix.shape[1] / SR, "s")


if __name__ == "__main__":
    main()
