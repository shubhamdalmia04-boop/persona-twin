# Persona Twin

> **Use responsibly.** This builds an AI simulation of a specific, real person from their private
> messages, recordings and photos. Only build one with that person's consent, or — if they have
> died — as someone with a legitimate, family-recognised right to do so. The same technique can be
> used to impersonate people without their consent; don't do that, and don't use this to deceive
> anyone about who or what they are talking to.

Build an AI simulation of a person from their messages, recordings, documents and photos, talk to
it by text or voice, and show it on a screen, a projector or a hologram pyramid. After a one-time
setup everything runs **offline** on your own computer.

## What do I run?

**Windows users (the packaged app):** unzip `PersonaTwin-windows.zip` and double-click
**`PersonaTwin.exe`**. It has three tabs: *Set up*, *Build a twin*, *Talk*. See `README-FIRST.txt`
for the step-by-step guide. Nobody needs to install Python.

**Building that zip (once, by whoever distributes it).** A Windows `.exe` can only be built on
Windows, so pick one:

* On a Windows PC with Python 3.12: double-click `build_windows.bat`
  (or run `py -3.12 build_exe.py`). Result: `dist/PersonaTwin-windows.zip`.
* Without a Windows PC: push this folder to GitHub and run the **Build Windows app** workflow
  (`.github/workflows/build-windows.yml`); download the `PersonaTwin-windows` artifact.

**From source (Windows, macOS, Linux):** `python launcher/main.py` opens the same window
(Linux needs `sudo apt install python3-tk`). Power users can skip the window and use the
command line below.

The exe is a small launcher (about 10 MB). "Set up" downloads a private Python and the packages
next to it, so the exe itself never has to bundle gigabytes of AI libraries. Everything lives in
the folders beside the exe: `runtime\` (Python + packages), `data\` (your twins), `app\` (program
files). Delete the folder to remove everything.

## Choosing where their material is

On the *Build a twin* tab press **Choose folder...** and pick any folder on the computer. The
folder and all its sub-folders are searched (system and hidden folders are skipped), and a
summary is shown before anything is processed. Supported:

| Found in the folder | Used for |
|---|---|
| WhatsApp `.txt` exports, Telegram/Facebook-style `.json` chats | memories + writing style (only the chosen person's messages) |
| other `.txt` / `.md`, Word `.docx`, text PDFs | memories (treated as written by the person) |
| audio (`.mp3 .m4a .opus .wav ...`) and video (`.mp4 .mov ...`) | transcribed offline into memories; the cleanest speech becomes the voice reference |
| photos (`.jpg .png .webp`, iPhone `.heic`) | avatar candidates (up to 30 are kept) |
| `about.txt` | facts you want it to know (relationship, birthplace, habits) |

Things worth knowing:

* **Pick the person's name from the chats.** The window lists who wrote the most messages so you
  can choose the right one.
* **Choose one clear face photo.** Otherwise a photo from the folder is guessed, and it might be
  someone else. Best: front-facing, mouth closed, eyes open, good light.
* **Recordings are assumed to be only this person.** Other voices get mixed into the twin, so use a
  folder that is theirs, or untick "use audio and video". Long recordings are limited by a time
  budget (default 90 minutes, short voice notes first) because transcription on a laptop is slow.
* Scanned PDFs (pictures of pages) have no text and are skipped.

## Making it feel personal

Chats are mostly "on my way", "ok see you" — small talk with little of *who someone was*. A twin
built from chats alone will sound generic. Three things make the biggest difference, all on the
*Build a twin* tab:

* **Write notes about them.** The "Notes about them" box is the single biggest lever. A few
  sentences of nicknames, sayings, stories, opinions and how they greeted people go a lot further
  than another thousand "ok"s. For example: *"Called me 'chhoti' since I was five. Always said
  'khaana khaya?' before anything else. Loved reminiscing about his college cricket team. Never
  left an argument without the last word, but always brought tea after."* The twin can only state
  facts it has — if it isn't written down or in the chats, it will say so rather than invent it.
* **Say who is talking to it.** "Who will be talking to them?" (for example *"his daughter
  Meera"*) changes how it answers — the same way a person answers their child differently than a
  stranger. Set it again per session on the *Talk* tab if several people will use it.
* **Give it real chat history**, not just voice notes. Real exchanges are shown to the model as
  example turns of the conversation, which is what actually teaches it the person's manner of
  speaking — short or long replies, teasing, bluntness, which language they switched to and when.

If it still doesn't sound like them, press **"About this twin"** on the *Talk* tab. It reports
how many messages actually matched the person's name (a wrong name here means the chats taught it
nothing), how many are kept as tone examples, whether a voice and a face photo were found, and
what to add next. Tick **"Show what it remembered"** while chatting to see, message by message,
which memories and example exchanges it used to answer — useful for telling whether the person's
name was spelled differently in some chats, or whether a topic just never came up in the source
material.

## Languages

On the *Build a twin* tab, "Language they speak" sets the language for three things at once:
transcribing their recordings, how the cloned voice speaks, and the language replies are given
in. It is saved with the twin, so you only set it once. Pick **"Hindi + English mixed
(Hinglish)"** if your father mixed languages the way many families do — the twin is also told to
copy whatever mixing pattern shows up in his real messages, so if he wrote half in Hindi script
and half in Latin-script Hindi/English, it will tend to do the same.

A few things affect how well this works in practice:

* **The language model (the "brain") matters most.** The default model is a general-purpose one
  and its Hindi is workable but not as fluent as its English. For Hindi specifically, Llama 3.1
  officially supports it and tends to do noticeably better. When you pick Hindi, the app shows a
  **"Download now"** button for it (about 4.7 GB, one-time, needs the internet) and a
  **"Use it for this twin"** checkbox, which is left off by default so picking the language alone
  never makes an existing twin stop working. Downloading ticks the checkbox automatically once it
  finishes; if the model was already installed in an earlier session, the checkbox is ticked
  straight away with no download needed.
* **Speech recognition (Whisper)** handles Hindi well, script or transliteration either way, and
  the multilingual memory-search model works across languages without any setup.
  Recordings that switch language mid-sentence are the hardest case for it; setting the language
  reduces mistaken guesses.
* **The cloned voice (XTTS)** speaks 17 languages, Hindi included. Devanagari text always
  switches it to Hindi automatically, even if the language wasn't set; setting the language also
  covers Hindi written in Latin script.
* **More Hindi chat history helps more than anything else here.** The examples shown to the model
  are real exchanges pulled from the chats — the more of those there are in Hindi, the more
  naturally it answers in Hindi. "About this twin" reports how many were used.

For a language not in the list, use the command line: `--language` accepts any Whisper code, and
`--llm-model`/`--tts` let you pick a different language model or fall back to the generic voice
for a language XTTS doesn't cover (`python run.py languages` lists the codes the app knows).

## Command line (advanced)

```
python install.py                                        # one-time setup into a .venv
python run.py ingest --name Asha --path D:\Family --speaker "Asha Verma" --face-photo asha.jpg \
  --about-file asha_notes.txt --listener "her daughter Priya"
python run.py chat   --name Asha --display single        # talk; projector / window
python run.py chat   --name Asha --listener "her daughter Priya" --show-memories
python run.py chat   --name Papa --language hi --llm-model llama3.1:8b
python run.py inspect --name Asha                          # what it learned, what is missing
python run.py languages                                    # list language codes
python run.py chat   --name Asha --voice --display pyramid
python run.py voice  --name Asha --say "Hello, how are you?"
python run.py list
```

Useful flags: `--windowed`, `--screen 1` (second monitor/projector), `--mirror`, `--no-speech`,
`--tts system` (fast generic voice), `--avatar photo` (simple photo instead of the animated face),
`--no-media`, `--media-minutes 30`. Press Esc or Q to close the picture.

## The face

The display shows an animated talking face built from one photo, designed to run in real time on
an ordinary CPU (about 2 ms per frame at the default 384 px). It opens the jaw and widens or rounds
the lips in time with the voice, blinks, sways slightly and paints a mouth interior when the mouth
opens. It is a 2D puppet, not photoreal lip-sync. If MediaPipe or its model is missing, or no usable
face is found, it falls back to the photo with a glow and says why. `TWIN_FACE_SIZE=256` helps on a
very slow machine.

## The person's voice

`ingest` keeps the cleanest ~12 second stretches of speech from the recordings as reference clips,
and `chat` speaks in that voice (Coqui XTTS-v2, optional install). Without clips or without voice
cloning it uses a generic system voice.

```
python run.py voice --name Asha                          # list stored reference clips
python run.py voice --name Asha --add call.m4a note.opus # add your own (preferred over automatic)
python run.py voice --name Asha --clear
```

20-60 seconds of clean single-speaker speech works best. XTTS-v2 speaks 17 languages
(`TWIN_TTS_LANGUAGE`, default `en`; Devanagari text switches to Hindi). Its licence is
**non-commercial**: fine for personal use, not for selling this as a product.

## Projection

* **Projector on a wall/screen:** "Fullscreen / projector". Black is not projected, so use a dark room.
* **Floating look:** project onto holographic rear-projection film or gauze, with "Mirror" if
  projecting from behind.
* **Pepper's ghost pyramid:** place a 4-sided acrylic pyramid, small end down, in the middle of a
  flat screen or tablet and choose "Hologram pyramid". If the picture looks reversed, tick "Mirror".

## How it works

```
mic -> Whisper -> memory search -> local LLM (Ollama) -> sentence -> cloned voice -> speaker
                                                                              \-> mouth shapes -> animated face
       built from chats, documents, recordings, about.txt
```

Replies are grounded in retrieved memories. The model is told to say "I don't remember" instead of
inventing events, and to admit it is an AI if sincerely asked. The display always shows an
"AI simulation" label. A hardware check at setup picks model sizes that stay usable (a typical
laptop gets a 3B language model and the `base` speech model; see `data/profile.json`).

## Current limits

* **Untested on real hardware by the author:** the tests cover the logic, but the Windows exe build,
  the download/setup flow, the window, the MediaPipe face detection and the XTTS voice have not been
  run end to end. Expect small fixes on the first real run.
* **The face is a 2D puppet.** It cannot turn its head, change expression or match exact phonemes.
* **On a laptop without a GPU the cloned voice is slow** (a few seconds per sentence) and the
  language model answers at a few words per second. `Generic voice (fast)` helps.
* The unsigned exe triggers Windows SmartScreen ("More info" > "Run anyway") and some antivirus
  tools flag PyInstaller files; code signing fixes that.
* Small local models can drift in character. More messages and a good `about.txt` help.

## Consent and safety

Building a twin asks you to confirm you are that person, have their permission, or are legally and
ethically entitled to use their material (for example as family). Keep it: this data is very
sensitive and the same pipeline could be used to impersonate someone. `data\` holds everything
about each twin; protect it (disk encryption) and delete a twin from the Talk tab to remove it.

## Configuration

Environment variables: `TWIN_LLM_MODEL`, `TWIN_EMBED_MODEL`, `TWIN_WHISPER_MODEL`,
`TWIN_WHISPER_LANGUAGE`, `TWIN_TTS_LANGUAGE`, `TWIN_FACE_SIZE`, `TWIN_MEDIA_MINUTES`, `OLLAMA_URL`,
`TWIN_DATA_DIR`.

## Tests

```
python -m unittest
```

## License

The code in this repository is MIT-licensed — see [LICENSE](LICENSE). Third-party models the app
downloads on first use (voice cloning, language models, the face model) are each under their own
publisher's license, not this one; LICENSE lists them. Contributions are welcome via pull request.

