"""
ASCII audio-reactive video generator (v2 - lyrics driven).
Celá obrazovka vyplněná textem písně. Aktivní slovo (právě zpívané) zazáří.
Ostatní stopy modulují pozadí decentně.
"""
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import soundfile as sf
import multiprocessing as mp
from pathlib import Path
import os
import sys
import re

# ============ CONFIG ============

_BASE = os.path.dirname(os.path.abspath(__file__))

TRACKS = [
    os.path.join(_BASE, "0 Lead Vocals.wav"),
    os.path.join(_BASE, "1 Backing Vocals.wav"),
    os.path.join(_BASE, "2 Drums.wav"),
    os.path.join(_BASE, "3 Bass.wav"),
    os.path.join(_BASE, "4 Percussion.wav"),
    os.path.join(_BASE, "5 Synth.wav"),
    os.path.join(_BASE, "6 Other.wav"),
]

# Mapping indexů do TRACKS na jednotlivé role ve vizuálu:
VOX_TRACK_INDEX   = 0  # Lead Vocals    -> driver pro lyrics + boost aktivního slova
PAD_TRACK_INDEX   = 1  # Backing Vocals -> background pulse
KICK_TRACK_INDEX  = 2  # Drums          -> primární onset flash
BASS_TRACK_INDEX  = 3  # Bass           -> globální boost CELÉ obrazovky
PERC_TRACK_INDEX  = 4  # Percussion     -> sekundární onset flash (sjednoceno s kick)
LEAD_TRACK_INDEX  = 5  # Synth          -> mírná globální modulace
OTHER_TRACK_INDEX = 6  # Other          -> přidává se do pad-like pozadí

OUTPUT_VIDEO = os.path.join(_BASE, "ascii_output.mp4")
OUTPUT_AUDIO = os.path.join(_BASE, "mixed.wav")
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
COLOR_DARK   = (10, 11, 14)      # filler / pozadí (skoro neviditelné)
COLOR_MID    = (95, 100, 108)    # tlumená šedá (běžný text)
COLOR_ACCENT = (220, 130, 55)    # NAŠE ORANŽOVÁ - aktivní slovo + flash
# Pokud máš jiný hex (třeba #E07F30), uprav řádek COLOR_ACCENT výš.
# Konverze hex->RGB: #RRGGBB -> (0xRR, 0xGG, 0xBB)
BG_COLOR = (6, 7, 10)

ACCENT_THRESHOLD = 0.55  # nad tento level intenzity se přepíná z šedé na oranžovou

FILLER_CHAR = '·'
FILLER_BASE_INTENSITY = 0.08

# === Citlivost (gain pro každou stopu) ===
GAIN_PAD          = 1.6    # pad RMS → background pulse
GAIN_KICK         = 0.32   # kick onset → flash
KICK_THRESHOLD    = 0.45   # kick onset threshold
GAIN_BASS_GLOBAL  = 0.55   # BASS RMS → globální boost CELÉ obrazovky (silný!)
GAIN_LEAD         = 0.10   # lead RMS → globální mírná modulace
GAIN_VOX_BOOST    = 0.40   # vox RMS → boost aktivních slov
VOX_THRESHOLD     = 0.12   # pod tuto hladinu se aktivní slovo nerozsvítí (nic se nezpívá)
SMOOTH_ALPHA      = 0.55   # temporal smoothing

# === Horizontální scroll textu ===
SCROLL_ENABLED       = True
SCROLL_SPEED_MIN     = 0.10   # znaků za frame na nejpomalejších řádcích
SCROLL_SPEED_MAX     = 1.5    # znaků za frame na nejrychlejších (1/3 stránky)
SCROLL_PEAK_FRACTION = 0.33   # kde je vrchol vlny (0=top, 1=bottom)
SCROLL_REVERSE_EVERY = 6      # každý N-tý řádek jede opačně (1-indexed)

# === Per-row velikosti písma ===
# Každý řádek má svou velikost fontu - vytváří chaotickou typografickou kompozici.
# Větší font = řádek zabírá víc vertikální plochy a má v sobě méně znaků.
FONT_SIZE_ENABLED   = True
FONT_SIZE_BASE      = 18    # default velikost (jako předtím)
FONT_SIZE_MIN       = 12    # minimální velikost na řádku
FONT_SIZE_MAX       = 32    # maximální velikost (pozor: moc velké rozbije layout)
# Pattern velikostí - řádky se cyklí přes tento seznam (custom rytmus typografie):
FONT_SIZE_PATTERN   = [16, 14, 22, 18, 13, 28, 15, 19, 14, 24, 17, 13]

# Lyrics timing:
#   "linear"     - slova rovnoměrně přes audio délku (ignoruje intro a pauzy - desync)
#   "vox_gated"  - slova postupují jen když vokál hraje, v pauzách drží (DOPORUČENO)
#   "vox_onset"  - postup po vokálových onsetech (přesný ale citlivý na detekci)
LYRICS_TIMING = "vox_gated"

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

# ============ AUDIO + FEATURES ============

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    print("VAROVÁNÍ: monospace font nenalezen, default PIL font")
    return None

def extract_features(path, sr, hop):
    print(f"  načítám {path}")
    y, _ = librosa.load(path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=hop*2)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = (rms / (rms.max() + 1e-9)).astype(np.float32)
    onset = (onset / (onset.max() + 1e-9)).astype(np.float32)
    return {'y': y.astype(np.float32), 'rms': rms, 'onset': onset}

# ============ LYRICS PARSING + LAYOUT ============

def parse_lyrics(text):
    text = re.sub(r'\[[^\]]*\]', ' ', text)        # brackets pryč
    text = text.replace('\u2026', '...')           # … → ...
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = re.sub(r'\s+', ' ', text).strip()
    return [w for w in text.split(' ') if w]

def layout_lyrics_grid(words, cols, rows, filler_char):
    """
    Vyplní celý grid slovy (cyklické opakování pokud je text krátký).
    Vrací (char_grid, positions).
    """
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
    """Vrací index do positions[] - kurzor postupuje sekvenčně gridem."""
    if n_positions == 0:
        return np.zeros(n_frames, dtype=np.int32)
    if mode == "linear":
        return ((np.arange(n_frames) / max(1, n_frames - 1)) * (n_positions - 1)).astype(np.int32)
    elif mode == "vox_gated":
        # Postupuje jen když vokál hraje, v pauzách drží
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
    """
    Per-frame intensity grid.
    - Bass = silný globální boost CELÉ obrazovky
    - Kurzor postupuje sekvenčně textem (active_pos)
    - V každém framu rozsvítí VŠECHNY výskyty aktuálního slova v gridu
    - Plus trail posledních N slov (taky všechny jejich výskyty)
    """
    print("Počítám intenzitní pole...")

    def _norm(w):
        return w.lower().strip(".,;:!?'\"()[]")

    # Mapa: text slova -> seznam všech (r, c0, c1) výskytů v gridu
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
    trail_len = 4  # kolik předchozích slov ještě dosvěcuje (krátký trail)

    for fi in range(n_frames):
        target = base_filler.copy()

        # Pad - background pulse (Backing Vocals + Other dohromady)
        pad_e = max(
            float(features[PAD_TRACK_INDEX]['rms'][fi]),
            float(features[OTHER_TRACK_INDEX]['rms'][fi]) * 0.7,
        )
        target += pad_field * pad_e * GAIN_PAD

        # BASS - silný globální boost CELÉ obrazovky
        bass_e = float(features[BASS_TRACK_INDEX]['rms'][fi])
        target += bass_e * GAIN_BASS_GLOBAL

        # Lead (Synth) - mírná globální modulace
        lead_e = float(features[LEAD_TRACK_INDEX]['rms'][fi])
        target += lead_e * GAIN_LEAD

        # Kick - krátký flash na onsetu (max z Drums a Percussion)
        kick_onset = max(
            float(features[KICK_TRACK_INDEX]['onset'][fi]),
            float(features[PERC_TRACK_INDEX]['onset'][fi]),
        )
        if kick_onset > KICK_THRESHOLD:
            target += (kick_onset - KICK_THRESHOLD) * GAIN_KICK

        # KURZOR + AKTUÁLNÍ SLOVO + trail - jen když vokál hraje
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
                # Slovo na této pozici
                word_text = _norm(words[positions[pi][0]])
                # Rozsvítit VŠECHNY jeho výskyty v celém gridu
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
    """Per-řádek velikost fontu (v px). Cyklí přes FONT_SIZE_PATTERN."""
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
    """Y pozice (top) každého řádku v pixelech.
    Roztáhneme/stáhneme aby suma seděla na HEIGHT."""
    raw_heights = [int(s * 1.05) for s in row_sizes]  # 1.05 = line spacing
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
    """Vlnitý profil rychlostí: nejrychlejší v SCROLL_PEAK_FRACTION řádku.
    Každý SCROLL_REVERSE_EVERY-tý řádek jede opačným směrem (záporný speed)."""
    speeds = np.zeros(ROWS, dtype=np.float32)
    peak_row = SCROLL_PEAK_FRACTION * (ROWS - 1)
    # Vzdálenost od peaku, normalizovaná na max vzdálenost
    max_dist = max(peak_row, (ROWS - 1) - peak_row)
    for r in range(ROWS):
        d = abs(r - peak_row) / max(1.0, max_dist)
        # Cosine falloff od peaku
        falloff = 0.5 * (1 + np.cos(np.pi * min(1.0, d)))  # 1 na peaku, 0 na okrajích
        speed = SCROLL_SPEED_MIN + (SCROLL_SPEED_MAX - SCROLL_SPEED_MIN) * falloff
        # Každý 6. řádek (1-indexed) opačný směr
        if (r + 1) % SCROLL_REVERSE_EVERY == 0:
            speed = -speed
        speeds[r] = speed
    return speeds

def make_shifted_indices(row_speeds, frame_idx, cols):
    """Pro daný frame vrať per-row offset (int) do source gridu (modulo cols)."""
    # offset = round(speed * frame_idx) mod cols
    offsets = (row_speeds * frame_idx).astype(np.int64) % cols
    return offsets.astype(np.int32)


# ============ FRAME -> IMAGE ============

def _lerp_color(lo, hi, t):
    return (int(lo[0] + (hi[0] - lo[0]) * t),
            int(lo[1] + (hi[1] - lo[1]) * t),
            int(lo[2] + (hi[2] - lo[2]) * t))

N_LEVELS = 32

def _build_lut():
    """Trojbodový gradient: DARK -> MID -> ACCENT.
    Pod ACCENT_THRESHOLD lerp DARK->MID (monochrom šedá).
    Nad ACCENT_THRESHOLD lerp MID->ACCENT (oranžový highlight)."""
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
    """fonts: list[ROWS] of ImageFont. y_positions: list[ROWS] of int (top y)."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Scroll: per-row roluj jak intenzitu tak znaky o row_offsets[r]
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
        # Šířka znaku tohoto fontu (přibližně FONT_SIZE * 0.6 pro monospace bold)
        # Použiju .getlength pro přesnost:
        try:
            cw = font.getlength('M')
        except Exception:
            cw = font.size * 0.6
        # Kolik znaků se vejde na šířku
        n_chars = max(1, int(WIDTH / cw))
        # Vezmi prvních n_chars buněk z gridu (grid má COLS, pokud n_chars > COLS, ořež)
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

    # Per-row velikosti fontů
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
    missing = [t for t in TRACKS if not os.path.exists(t)]
    if missing:
        print(f"\n!!! Nenalezeno {len(missing)} stop. Hledám v: {_BASE}")
        try:
            for f in sorted(os.listdir(_BASE)):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    print(f"     {f!r}")
        except Exception as e:
            print(f"     (chyba listingu: {e})")
        for t in missing:
            print(f"     hledá: {os.path.basename(t)!r}")
        sys.exit(1)

    FRAMES_DIR.mkdir(exist_ok=True)
    hop = SR // FPS
    font_path = find_font()

    print("Extrakce features...")
    features = [extract_features(t, SR, hop) for t in TRACKS]

    print("Mix audio...")
    max_len = max(len(f['y']) for f in features)
    mixed = np.zeros(max_len, dtype=np.float32)
    for f in features:
        mixed[:len(f['y'])] += f['y']
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak * 0.9
    sf.write(OUTPUT_AUDIO, mixed, SR)

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
        "-i", OUTPUT_AUDIO,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        OUTPUT_VIDEO
    ], check=True)
    print(f"Hotovo: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
