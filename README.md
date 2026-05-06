# ASCII Song Visualiser

Audio-reactive ASCII video generátor. Vyplní celou obrazovku textem písně, aktivně "zpívané" slovo zazáří oranžově, ostatní stopy (kick, bass, pad, lead) modulují pozadí. Výstup: `1920×1080 @ 30 fps` MP4 s namixovaným zvukem.

![preview](preview.png)

Dvě verze:

| Skript | Vstup | Použití |
| --- | --- | --- |
| [`ascii_video_FINAL.py`](ascii_video_FINAL.py) | 5 oddělených stem stop (`kick`, `pad`, `bass`, `lead`, `vox`) | Hotová stopa, kde máš mix předem rozdělený |
| [`ascii_video_MIX.py`](ascii_video_MIX.py) | 1 kompletní mix (WAV/MP3/FLAC) | Když máš jen finální mix — skript si sám udělá HPSS + frekvenční split na 5 virtuálních stop |

## Vizuál

Aktivní slovo (kurzor postupuje gridem podle vokálu) září oranžově `(220, 130, 55)`, zbytek je tlumený monochrom `(95, 100, 108)`. Pozadí pulzuje basou a padem, kick dělá krátké flashe. Řádky se horizontálně skrolují s vlnitým profilem rychlostí, každý 6. řádek opačným směrem. Per-řádek velikosti fontu vytvářejí chaotickou typografickou kompozici.

## Požadavky

- **Python 3.9+**
- **ffmpeg** v `PATH` (pro finální slepení frames + audio do MP4)
- Python knihovny — viz [`requirements.txt`](requirements.txt):
  - `librosa` (extrakce features, HPSS, STFT)
  - `numpy`
  - `Pillow` (kreslení frames)
  - `soundfile` (jen `FINAL` verze — zápis namixovaného audia)

Instalace:

```bash
pip install -r requirements.txt
```

ffmpeg na Windows: stáhni z [ffmpeg.org](https://ffmpeg.org/download.html) a přidej do PATH. Na Linuxu `apt install ffmpeg`, na macOS `brew install ffmpeg`.

## Použití

### Verze MIX — jeden hotový mix

1. Polož finální mix vedle skriptu, pojmenuj `mix.wav` (nebo uprav `MIX_PATH` v hlavičce skriptu).
2. Uprav `LYRICS = """..."""` v hlavičce — vlož text písně. Hranaté závorky `[Intro]`, `[Drop]` atd. se ignorují.
3. Spusť:
   ```bash
   python ascii_video_MIX.py
   ```
4. Výstup: `ascii_output.mp4`.

Skript si z mixu vyrobí 5 "virtuálních stop":

| Virtuální stopa | Zdroj |
| --- | --- |
| kick | HPSS perkusivní složka, 20–150 Hz |
| bass | 60–300 Hz |
| pad | 300–1500 Hz |
| lead | 1500–4000 Hz |
| vox | HPSS harmonická složka, 200–3000 Hz |

### Verze FINAL — separátní stem stopy

1. Polož vedle skriptu pět WAV souborů:
   - `track1_kick.wav`
   - `track2_pad.wav`
   - `track3_bass.wav`
   - `track4_lead.wav`
   - `track1_vox.wav`
2. Uprav `LYRICS` v hlavičce.
3. Spusť:
   ```bash
   python ascii_video_FINAL.py
   ```
4. Výstup: `ascii_output.mp4` + `mixed.wav` (smíchané audio).

Pokud máš jiné názvy/struktury, uprav konstantu `TRACKS` na začátku skriptu.

## Konfigurace

Všechno se ladí v hlavičce skriptu (sekce `# ============ CONFIG ============`). Klíčové hodnoty:

### Rozlišení a tempo
```python
FPS = 30
WIDTH, HEIGHT = 1920, 1080
COLS, ROWS = 192, 54        # logický grid znaků
```

### Barvy
```python
COLOR_DARK   = (10, 11, 14)      # filler / pozadí
COLOR_MID    = (95, 100, 108)    # běžný text (tlumená šedá)
COLOR_ACCENT = (220, 130, 55)    # aktivní slovo (oranžová)
BG_COLOR     = (6, 7, 10)
ACCENT_THRESHOLD = 0.55          # nad tento level se přepíná z šedé na oranžovou
```

### Citlivost (gainy jednotlivých stop)
```python
GAIN_PAD          = 1.6     # pad → šum pozadí
GAIN_KICK         = 0.32    # kick onset → flash
KICK_THRESHOLD    = 0.45
GAIN_BASS_GLOBAL  = 0.55    # bass → globální boost CELÉ obrazovky
GAIN_LEAD         = 0.10    # lead → mírná modulace
GAIN_VOX_BOOST    = 0.40    # vox RMS → boost aktivního slova
VOX_THRESHOLD     = 0.12    # pod touto hladinou se aktivní slovo nerozsvítí
SMOOTH_ALPHA      = 0.55    # temporal smoothing (0=žádné, 1=maximální)
```

### Skrolování řádků
```python
SCROLL_ENABLED       = True
SCROLL_SPEED_MIN     = 0.10   # nejpomalejší řádky (znaky/frame)
SCROLL_SPEED_MAX     = 1.5    # nejrychlejší řádky
SCROLL_PEAK_FRACTION = 0.33   # kde je vrchol vlny (0=top, 1=bottom)
SCROLL_REVERSE_EVERY = 6      # každý N-tý řádek jede opačně
```

### Per-řádek velikosti fontu
```python
FONT_SIZE_PATTERN = [16, 14, 22, 18, 13, 28, 15, 19, 14, 24, 17, 13]
FONT_SIZE_MIN     = 12
FONT_SIZE_MAX     = 32
```

Pattern se cyklí přes všech 54 řádků. Větší font = řádek zabírá víc vertikální plochy a má v sobě méně znaků.

### Synchronizace lyrics
```python
LYRICS_TIMING = "vox_gated"     # doporučeno
```
- `linear` — slova rovnoměrně přes celou délku audia (ignoruje pauzy → desync na delších skladbách).
- `vox_gated` — kurzor postupuje jen když vokál hraje, v pauzách drží. **Doporučeno.**
- `vox_onset` — postup po jednotlivých onsetech vokálu (přesné, ale citlivé na detekci).

## Jak to vnitřně funguje

1. **Extrakce features** — pro každou stopu (nebo virtuální pásmo u MIX verze) se počítá per-frame RMS energie a onset strength.
2. **Lyrics layout** — text se rozparsuje na slova, vyplní se jím celý `192×54` grid (cyklicky pokud je text krátký), pamatují se pozice každého slova.
3. **Intenzitní pole** — pro každý frame se sestaví matice `(54, 192)` intenzit:
   - bass dělá globální boost,
   - pad dělá pulzující šumové pozadí,
   - kick onset přidá flash,
   - když vokál hraje, kurzor postupuje gridem a rozsvítí **všechny výskyty** aktuálního slova v gridu (plus krátký 4-slovní trail).
4. **Render** — multiprocessing pool vykreslí každý frame přes Pillow + monospace font, intenzitu mapuje na barvu přes 32-úrovňovou LUT (DARK → MID → ACCENT).
5. **ffmpeg** — frames + zvuk → H.264/AAC MP4.

Render je CPU-bound, paralelizuje se přes `mp.cpu_count() - 1` workerů. Frames se ukládají na disk do `frames/` (smaž si je po renderu, MP4 už je nepotřebuje).

## Tipy

- **První spuštění je pomalé** kvůli HPSS/STFT extrakci. U dlouhých skladeb (5+ min) počítej s několika minutami jen na features.
- **Disk space** — frames v `1920×1080` PNG zaberou cca 200–500 MB na minutu videa. Po `ffmpeg` slepení je smaž.
- **Když "aktivní slovo" desynchronizuje**, zkus `LYRICS_TIMING = "vox_onset"` nebo zvyš `VOX_THRESHOLD`. Pro MIX verzi pomáhá, když je vokál v mixu výrazný v pásmu 200–3000 Hz.
- **Když je všechno moc šedé/tmavé**, sniž `ACCENT_THRESHOLD` (např. na `0.45`) nebo zvyš `GAIN_BASS_GLOBAL`.
- **Když je všechno moc oranžové**, opačně.

## Licence

[MIT](LICENSE)
