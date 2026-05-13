"""
ASCII audio-reactive video generator (v3 - single-mix edition).

Stejný vizuál jako ascii_video_FINAL.py, ale místo 5 stem stop bere
JEDEN hlavní mix (stereo nebo mono WAV) a sám si z něj vytvoří
5 "virtuálních stop" frekvenčním rozdělením + HPSS:

  kick  = perkusivní složka (HPSS), low band 20-150 Hz
  bass  = 60-300 Hz
  pad   = 300-1500 Hz
  lead  = 1500-4000 Hz
  vox   = harmonická složka, 200-3000 Hz (rozsah vokálu)

Stačí upravit MIX_PATH níž a spustit.
"""
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import multiprocessing as mp
from pathlib import Path
import os
import sys
import re

# ============ CONFIG ============

_BASE = os.path.dirname(os.path.abspath(__file__))

# JEDINÝ vstup - hlavní mix (mono nebo stereo WAV/MP3/FLAC).
MIX_PATH = os.path.join(_BASE, "mix.wav")

OUTPUT_VIDEO = os.path.join(_BASE, "ascii_output.mp4")
FRAMES_DIR   = Path(os.path.join(_BASE, "frames"))

FPS = 30
WIDTH, HEIGHT = 1920, 1080
COLS, ROWS = 192, 54
SR = 44100

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]
FONT_SIZE = 18

# === Paleta: monochrom šedá + oranžový accent na aktivních slovech ===
COLOR_DARK   = (10, 11, 14)
COLOR_MID    = (95, 100, 108)
COLOR_ACCENT = (220, 130, 55)
BG_COLOR = (6, 7, 10)

ACCENT_THRESHOLD = 0.55

FILLER_CHAR = '·'
FILLER_BASE_INTENSITY = 0.08

# === Citlivost (gain pro každou virtuální stopu) ===
GAIN_PAD          = 1.6
GAIN_KICK         = 0.32
KICK_THRESHOLD    = 0.45
GAIN_BASS_GLOBAL  = 0.55
GAIN_LEAD         = 0.10
GAIN_VOX_BOOST    = 0.40
VOX_THRESHOLD     = 0.12
SMOOTH_ALPHA      = 0.55

# === Horizontální scroll textu ===
SCROLL_ENABLED       = True
SCROLL_SPEED_MIN     = 0.10
SCROLL_SPEED_MAX     = 1.5
SCROLL_PEAK_FRACTION = 0.33
SCROLL_REVERSE_EVERY = 6

# === Per-row velikosti písma ===
FONT_SIZE_ENABLED   = True
FONT_SIZE_BASE      = 18
FONT_SIZE_MIN       = 12
FONT_SIZE_MAX       = 32
FONT_SIZE_PATTERN   = [16, 14, 22, 18, 13, 28, 15, 19, 14, 24, 17, 13]

# Lyrics timing:
#   "linear"     - slova rovnoměrně přes audio délku
#   "vox_gated"  - postup jen když "vox band" hraje (DOPORUČENO i pro mix)
#   "vox_onset"  - postup po onsetech ve vox bandu
LYRICS_TIMING = "vox_gated"

# Index virtuální stopy "vox" (zachovává API původního skriptu)
VOX_TRACK_INDEX = 4

LYRICS = """
[Intro]
[syncopated kick drum, palm-muted electric guitar riff]

[Verse 1]
[breathy female vocals]
Bottom of the room where the shadows don't move, and they stay, and they stay, and they stay. Counting the lines in my head, but they blur, and they fade, and they fade into gray.

[layered vocal harmonies enter]
Bottom of the room.
B-bottom of the room where the shadows don't move, and they stay, and they stay, and they stay.

Bottom of the room where the shadows don't move, and they stay, and they stay, and they stay. Counting the lines in my head, but they blur, and they fade, and they fade into gray.

Bottom of the room where the shadows don't move, and they stay, and they stay, and they stay.

[Chorus]
[full electronic beat, driving bass]
No heart, no pulse, just a shape in the dark, and it's pulling me under somewhere. Through the wires to my mind, something calling, but I can't define. No voice, no sound, just a trace going around, and around, and around.

[Outro]
[atmospheric synth pads swell]
Bottom of the room.

B-bottom of the room where the shadows don't move, and they stay, and they stay, and they stay.

Bottom of the room where the shadows don't move, and they stay, and they stay, and they stay. Counting the lines in my head, but they blur, and they fade, and they fade into gray.

Bottom of the room where the shadows don't move, and they stay, and they stay, and they stay.
"""

# ============ AUDIO + FEATURES (single mix → 5 virtuálních stop) ============

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    print("VAROVÁNÍ: monospace font nenalezen, default PIL font")
    return None

def _normalize(x):
    return (x / (x.max() + 1e-9)).astype(np.float32)

def extract_features_from_mix(path, sr, hop):
    """
    Z jednoho mixu vytvoří 5 virtuálních stop (kick, pad, bass, lead, vox)
    pomocí HPSS + frekvenčního rozdělení STFT spektra.

    Vrací list dictů kompatibilních s původním kódem:
        [{'rms', 'onset'}] x 5  + 'y' u prvního pro mix audio.
    """
    print(f"  načítám mix: {path}")
    y, _ = librosa.load(path, sr=sr, mono=True)

    print("  HPSS (perkusivní/harmonický split)...")
    y_h, y_p = librosa.effects.hpss(y, margin=2.0)

    print("  STFT band split...")
    n_fft = 2048
    S   = np.abs(librosa.stft(y,   n_fft=n_fft, hop_length=hop))
    S_h = np.abs(librosa.stft(y_h, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def band_rms(spec, low, high):
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            return np.zeros(spec.shape[1], dtype=np.float32)
        band = spec[mask]
        return np.sqrt(np.mean(band ** 2, axis=0)).astype(np.float32)

    # RMS po pásmech
    kick_rms = band_rms(S,   20,  150)
    bass_rms = band_rms(S,   60,  300)
    pad_rms  = band_rms(S,  300, 1500)
    lead_rms = band_rms(S, 1500, 4000)
    vox_rms  = band_rms(S_h, 200, 3000)  # harmonická složka = čistší vokál

    # Onsety
    kick_onset = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=hop)
    gen_onset  = librosa.onset.onset_strength(y=y,   sr=sr, hop_length=hop)

    # Vox onset z harmonické složky v pásmu vokálu
    vox_mask = (freqs >= 200) & (freqs < 3000)
    if vox_mask.any():
        S_vox = S_h[vox_mask]
        vox_onset = librosa.onset.onset_strength(S=S_vox, sr=sr, hop_length=hop)
    else:
        vox_onset = gen_onset

    # Sjednotit délku (různé librosa funkce občas vrátí ±1 frame)
    n = min(len(kick_rms), len(bass_rms), len(pad_rms), len(lead_rms),
            len(vox_rms), len(kick_onset), len(gen_onset), len(vox_onset))
    def _trim(a):
        return a[:n]

    feats = [
        # 0 - kick
        {'rms': _normalize(_trim(kick_rms)), 'onset': _normalize(_trim(kick_onset))},
        # 1 - pad
        {'rms': _normalize(_trim(pad_rms)),  'onset': _normalize(_trim(gen_onset))},
        # 2 - bass
        {'rms': _normalize(_trim(bass_rms)), 'onset': _normalize(_trim(gen_onset))},
        # 3 - lead
        {'rms': _normalize(_trim(lead_rms)), 'onset': _normalize(_trim(gen_onset))},
        # 4 - vox
        {'rms': _normalize(_trim(vox_rms)),  'onset': _normalize(_trim(vox_onset))},
    ]
    # 'y' (čisté audio) připojím k prvnímu featuru, použije se pro délku videa
    feats[0]['y'] = y.astype(np.float32)
    return feats

# ============ LYRICS PARSING + LAYOUT ============

def parse_lyrics(text):
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = text.replace('…', '...')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text).strip()
    return [w for w in text.split(' ') if w]

def layout_lyrics_grid(words, cols, rows, filler_char):
    char_grid = [[filler_char] * cols for _ in range(rows)]
    positions = []
    if not words:
        return char_grid, positions
    r, c = 0, 0
    cyc = 0
    safety = rows * cols
    while r < rows and safety > 0:
        safety -= 1
        word = words[cyc % len(words)]
        wlen = len(word)
        if wlen > cols:
            word = word[:cols]
            wlen = cols
        if c + wlen > cols:
            r += 1
            c = 0
            continue
        for i, ch in enumerate(word):
            char_grid[r][c + i] = ch
        positions.append((cyc % len(words), r, c, c + wlen))
        c += wlen + 2
        cyc += 1
    return char_grid, positions

def compute_active_position_per_frame(features, n_frames, n_positions, mode):
    if n_positions == 0:
        return np.zeros(n_frames, dtype=np.int32)
    if mode == "linear":
        return ((np.arange(n_frames) / max(1, n_frames - 1)) * (n_positions - 1)).astype(np.int32)
    elif mode == "vox_gated":
        vox_rms = features[VOX_TRACK_INDEX]['rms'][:n_frames]
        threshold = max(0.08, float(np.percentile(vox_rms, 55)))
        is_singing = (vox_rms > threshold).astype(np.float32)
        cum = np.cumsum(is_singing)
        total = cum[-1] if cum[-1] > 0 else 1.0
        idx = (cum / total * (n_positions - 1)).astype(np.int32)
        return np.clip(idx, 0, n_positions - 1)
    elif mode == "vox_onset":
        vox_onset = features[VOX_TRACK_INDEX]['onset'][:n_frames]
        thr = max(0.25, float(np.percentile(vox_onset, 85)))
        is_onset = vox_onset > thr
        for i in range(1, len(is_onset)):
            if is_onset[i] and is_onset[i-1]:
                is_onset[i] = False
        idx = np.cumsum(is_onset) - 1
        idx = np.clip(idx, 0, n_positions - 1)
        return idx.astype(np.int32)
    raise ValueError(f"Unknown LYRICS_TIMING: {mode}")

# ============ INTENZITNÍ MAPA ============

def compute_intensity_volume(features, n_frames, positions, words, active_pos_array):
    print("Počítám intenzitní pole...")

    def _norm(w):
        return w.lower().strip(".,;:!?'\"()[]")

    word_text_positions = {}
    for orig_idx, r, c0, c1 in positions:
        text = _norm(words[orig_idx])
        if text:
            word_text_positions.setdefault(text, []).append((r, c0, c1))

    base_filler = np.full((ROWS, COLS), FILLER_BASE_INTENSITY, dtype=np.float32)

    rng = np.random.RandomState(42)
    pad_field = rng.random((ROWS, COLS)).astype(np.float32) * 0.15

    volume = np.zeros((n_frames, ROWS, COLS), dtype=np.float32)
    n_positions = len(positions)
    trail_len = 4

    for fi in range(n_frames):
        target = base_filler.copy()

        pad_e = float(features[1]['rms'][fi])
        target += pad_field * pad_e * GAIN_PAD

        bass_e = float(features[2]['rms'][fi])
        target += bass_e * GAIN_BASS_GLOBAL

        lead_e = float(features[3]['rms'][fi])
        target += lead_e * GAIN_LEAD

        kick_onset = float(features[0]['onset'][fi])
        if kick_onset > KICK_THRESHOLD:
            target += (kick_onset - KICK_THRESHOLD) * GAIN_KICK

        vox_e = float(features[VOX_TRACK_INDEX]['rms'][fi])
        if vox_e > VOX_THRESHOLD and n_positions > 0:
            cursor = int(active_pos_array[fi])
            for d in range(-trail_len, 1):
                pi = cursor + d
                if pi < 0 or pi >= n_positions:
                    continue
                if d == 0:
                    inten = min(1.0, 0.65 + vox_e * GAIN_VOX_BOOST)
                else:
                    inten = max(0.18, 0.70 + d * 0.13)
                word_text = _norm(words[positions[pi][0]])
                for r, c0, c1 in word_text_positions.get(word_text, []):
                    target[r, c0:c1] = np.maximum(target[r, c0:c1], inten)

        volume[fi] = target

    volume = volume.clip(0, 1)

    print("Smoothing...")
    smoothed = volume.copy()
    for fi in range(1, n_frames):
        smoothed[fi] = SMOOTH_ALPHA * volume[fi] + (1 - SMOOTH_ALPHA) * smoothed[fi - 1]
    return smoothed

# ============ PER-ROW FONTS ============

def compute_row_font_sizes():
    if not FONT_SIZE_ENABLED:
        return [FONT_SIZE_BASE] * ROWS
    pattern = FONT_SIZE_PATTERN if FONT_SIZE_PATTERN else [FONT_SIZE_BASE]
    sizes = []
    for r in range(ROWS):
        s = pattern[r % len(pattern)]
        s = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, s))
        sizes.append(s)
    return sizes

def compute_row_y_positions(row_sizes):
    raw_heights = [int(s * 1.05) for s in row_sizes]
    total = sum(raw_heights)
    scale = HEIGHT / total if total > 0 else 1.0
    heights = [h * scale for h in raw_heights]
    y_positions = []
    y = 0.0
    for h in heights:
        y_positions.append(int(y))
        y += h
    return y_positions, [int(h) for h in heights]


# ============ SCROLL MECHANIKA ============

def compute_row_speeds():
    speeds = np.zeros(ROWS, dtype=np.float32)
    peak_row = SCROLL_PEAK_FRACTION * (ROWS - 1)
    max_dist = max(peak_row, (ROWS - 1) - peak_row)
    for r in range(ROWS):
        d = abs(r - peak_row) / max(1.0, max_dist)
        falloff = 0.5 * (1 + np.cos(np.pi * min(1.0, d)))
        speed = SCROLL_SPEED_MIN + (SCROLL_SPEED_MAX - SCROLL_SPEED_MIN) * falloff
        if (r + 1) % SCROLL_REVERSE_EVERY == 0:
            speed = -speed
        speeds[r] = speed
    return speeds

def make_shifted_indices(row_speeds, frame_idx, cols):
    offsets = (row_speeds * frame_idx).astype(np.int64) % cols
    return offsets.astype(np.int32)


# ============ FRAME -> IMAGE ============

def _lerp_color(lo, hi, t):
    return (int(lo[0] + (hi[0] - lo[0]) * t),
            int(lo[1] + (hi[1] - lo[1]) * t),
            int(lo[2] + (hi[2] - lo[2]) * t))

N_LEVELS = 32

def _build_lut():
    lut = []
    for i in range(N_LEVELS):
        t = i / (N_LEVELS - 1)
        if t < ACCENT_THRESHOLD:
            tt = t / ACCENT_THRESHOLD
            lut.append(_lerp_color(COLOR_DARK, COLOR_MID, tt))
        else:
            tt = (t - ACCENT_THRESHOLD) / (1.0 - ACCENT_THRESHOLD)
            lut.append(_lerp_color(COLOR_MID, COLOR_ACCENT, tt))
    return lut

COLOR_LUT = _build_lut()

def render_frame_image(intensity, char_grid, fonts, y_positions, char_w, row_offsets):
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    if row_offsets is not None:
        rolled_intensity = np.empty_like(intensity)
        rolled_chars = []
        for r in range(ROWS):
            off = int(row_offsets[r])
            rolled_intensity[r] = np.roll(intensity[r], -off)
            rc = char_grid[r]
            rolled_chars.append(rc[off % COLS:] + rc[:off % COLS])
        intensity = rolled_intensity
        char_grid = rolled_chars

    levels = (intensity * (N_LEVELS - 1)).astype(np.int32).clip(0, N_LEVELS - 1)

    for r in range(ROWS):
        y = y_positions[r]
        font = fonts[r]
        try:
            cw = font.getlength('M')
        except Exception:
            cw = font.size * 0.6
        n_chars = max(1, int(WIDTH / cw))
        n_chars = min(n_chars, COLS)
        row_levels = levels[r][:n_chars]
        row_chars = char_grid[r][:n_chars]

        c = 0
        while c < n_chars:
            lvl = int(row_levels[c])
            run_start = c
            while c < n_chars and int(row_levels[c]) == lvl:
                c += 1
            text = ''.join(row_chars[run_start:c])
            x = int(run_start * cw)
            draw.text((x, y), text, fill=COLOR_LUT[lvl], font=font)
    return img

# ============ WORKER (multiprocessing) ============

_W = {}

def _init_worker(volume_path, char_grid_path, font_path, frames_dir,
                 row_speeds_path, row_sizes_path):
    _W['volume'] = np.load(volume_path, mmap_mode='r')
    with open(char_grid_path, 'r', encoding='utf-8') as f:
        rows_text = f.read().split('\n')
    grid = [list(row.ljust(COLS)[:COLS]) for row in rows_text[:ROWS]]
    while len(grid) < ROWS:
        grid.append([FILLER_CHAR] * COLS)
    _W['char_grid'] = grid

    if row_sizes_path and os.path.exists(row_sizes_path):
        row_sizes = np.load(row_sizes_path).tolist()
    else:
        row_sizes = [FONT_SIZE_BASE] * ROWS

    fonts = []
    for s in row_sizes:
        if font_path:
            fonts.append(ImageFont.truetype(font_path, int(s)))
        else:
            fonts.append(ImageFont.load_default())
    _W['fonts'] = fonts
    _W['y_positions'], _ = compute_row_y_positions(row_sizes)
    _W['char_w'] = WIDTH / COLS
    _W['frames_dir'] = Path(frames_dir)
    if row_speeds_path:
        _W['row_speeds'] = np.load(row_speeds_path)
    else:
        _W['row_speeds'] = None

def _render_one(frame_idx):
    intensity = _W['volume'][frame_idx]
    if _W['row_speeds'] is not None:
        offsets = make_shifted_indices(_W['row_speeds'], frame_idx, COLS)
    else:
        offsets = None
    img = render_frame_image(intensity, _W['char_grid'],
                             _W['fonts'], _W['y_positions'],
                             _W['char_w'], offsets)
    img.save(_W['frames_dir'] / f"{frame_idx:06d}.png")
    return frame_idx

# ============ MAIN ============

def main():
    if not os.path.exists(MIX_PATH):
        print(f"\n!!! Mix nenalezen: {MIX_PATH}")
        print(f"   Hledám audio v: {_BASE}")
        try:
            for f in sorted(os.listdir(_BASE)):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    print(f"     {f!r}")
        except Exception as e:
            print(f"     (chyba listingu: {e})")
        print("\n   Uprav MIX_PATH na začátku skriptu.")
        sys.exit(1)

    FRAMES_DIR.mkdir(exist_ok=True)
    hop = SR // FPS
    font_path = find_font()

    print("Extrakce features z mixu...")
    features = extract_features_from_mix(MIX_PATH, SR, hop)

    n_frames = min(len(f['rms']) for f in features)
    print(f"Total framů: {n_frames} ({n_frames / FPS:.1f}s)")

    print("Lyrics layout...")
    words = parse_lyrics(LYRICS)
    print(f"  {len(words)} slov")
    char_grid, positions = layout_lyrics_grid(words, COLS, ROWS, FILLER_CHAR)
    print(f"  {len(positions)} pozic v gridu")

    active_pos = compute_active_position_per_frame(features, n_frames, len(positions), LYRICS_TIMING)
    print(f"  timing: {LYRICS_TIMING}, pojede {len(positions)} pozic")

    volume = compute_intensity_volume(features, n_frames, positions, words, active_pos)
    del features

    vol_path = os.path.join(_BASE, "_volume_tmp.npy")
    grid_path = os.path.join(_BASE, "_chargrid_tmp.txt")
    speeds_path = os.path.join(_BASE, "_speeds_tmp.npy")
    np.save(vol_path, volume)
    del volume
    with open(grid_path, 'w', encoding='utf-8') as f:
        for row in char_grid:
            f.write(''.join(row) + '\n')

    if SCROLL_ENABLED:
        row_speeds = compute_row_speeds()
        np.save(speeds_path, row_speeds)
        print(f"Scroll enabled. Top speed: {SCROLL_SPEED_MAX} ch/frame, "
              f"reverse každý {SCROLL_REVERSE_EVERY}. řádek")
        speeds_arg = speeds_path
    else:
        speeds_arg = None

    sizes_path = os.path.join(_BASE, "_sizes_tmp.npy")
    if FONT_SIZE_ENABLED:
        row_sizes = compute_row_font_sizes()
        np.save(sizes_path, np.array(row_sizes, dtype=np.int32))
        print(f"Per-row fonty: pattern {FONT_SIZE_PATTERN}, "
              f"min={FONT_SIZE_MIN} max={FONT_SIZE_MAX}")
        sizes_arg = sizes_path
    else:
        sizes_arg = None

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Render: pool {n_workers} workerů")

    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(vol_path, grid_path, font_path,
                           str(FRAMES_DIR), speeds_arg, sizes_arg)) as pool:
        done = 0
        for _ in pool.imap_unordered(_render_one, range(n_frames), chunksize=10):
            done += 1
            if done % 100 == 0 or done == n_frames:
                print(f"  {done}/{n_frames}  ({100*done/n_frames:.0f}%)")

    os.remove(vol_path)
    os.remove(grid_path)
    if SCROLL_ENABLED and os.path.exists(speeds_path):
        os.remove(speeds_path)
    if FONT_SIZE_ENABLED and os.path.exists(sizes_path):
        os.remove(sizes_path)

    print("ffmpeg slep...")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "%06d.png"),
        "-i", MIX_PATH,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ], check=True)
    print(f"Hotovo: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
