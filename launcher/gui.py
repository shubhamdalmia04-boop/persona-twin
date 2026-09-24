"""The Persona Twin window: set up once, pick a folder, build a twin, talk to it."""
from __future__ import annotations

import os
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from twin.config import slugify
from twin.hardware import total_ram_gb
from twin.hardware import recommend_llm_for_language, has_nvidia_gpu, is_apple_silicon
from twin.languages import DEFAULT_LANGUAGE, LANGUAGES, code_for, name_for
from twin.scan import scan_folder

from . import core

XTTS_TERMS = (
    "Voice cloning uses Coqui XTTS-v2 (about 2 GB).\n\n"
    "Its model licence (Coqui Public Model License) allows NON-COMMERCIAL use only. "
    "Personal use is fine; selling this or offering it as a paid service is not.\n\n"
    "Install voice cloning?"
)
CONSENT = (
    "I am this person, or I have their permission, or (if they have died) I am legally and "
    "ethically entitled to use their material, for example as their family."
)
VOICE_CHOICES = {
    "The person's voice (if available)": ("auto", True),
    "Generic voice (fast)": ("system", True),
    "Silent, text only": ("auto", False),
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Persona Twin")
        self.geometry("880x760")
        self.minsize(760, 620)
        self._ui: queue.Queue = queue.Queue()
        self.session = core.ChatSession()
        self.scan_speakers: list[tuple[str, int]] = []

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.setup_tab, self.build_tab, self.talk_tab = (ttk.Frame(self.tabs, padding=12) for _ in range(3))
        self.tabs.add(self.setup_tab, text="1. Set up")
        self.tabs.add(self.build_tab, text="2. Build a twin")
        self.tabs.add(self.talk_tab, text="3. Talk")

        self._build_setup()
        self._build_build()
        self._build_talk()
        self._refresh_state()
        self.after(100, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------- plumbing
    def _post(self, fn, *args) -> None:
        self._ui.put((fn, args))

    def _drain(self) -> None:
        try:
            while True:
                fn, args = self._ui.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _append(self, box: scrolledtext.ScrolledText, line: str) -> None:
        box.configure(state="normal")
        box.insert("end", line + "\n")
        box.see("end")
        box.configure(state="disabled")

    def _logger(self, box: scrolledtext.ScrolledText):
        return lambda line: self._post(self._append, box, line)

    def _log_box(self, parent) -> scrolledtext.ScrolledText:
        box = scrolledtext.ScrolledText(parent, height=12, state="disabled", wrap="word")
        box.pack(fill="both", expand=True, pady=(8, 0))
        return box

    def _run_bg(self, work, on_error_box: scrolledtext.ScrolledText | None = None) -> None:
        def wrapper() -> None:
            try:
                work()
            except Exception as exc:  # show any failure in the window instead of dying silently
                if on_error_box is not None:
                    self._post(self._append, on_error_box, f"ERROR: {exc}")
                self._post(messagebox.showerror, "Something went wrong", str(exc))
        threading.Thread(target=wrapper, daemon=True).start()

    def _close(self) -> None:
        self.session.stop()
        self.destroy()

    # ------------------------------------------------------------- tab 1: setup
    def _build_setup(self) -> None:
        f = self.setup_tab
        self.setup_status = ttk.Label(f, font=("", 12, "bold"))
        self.setup_status.pack(anchor="w")
        ttk.Label(
            f, wraplength=800, justify="left",
            text=("One-time setup downloads roughly 3 to 6 GB (speech, memory and language models, plus the tools "
                  "that run them). It needs the internet once. After that everything runs on this computer, "
                  "offline. Your files never leave this machine."),
        ).pack(anchor="w", pady=(6, 8))
        self.voice_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f, variable=self.voice_var,
            text="Also install voice cloning, so replies sound like the person (extra ~2 GB; slow without a GPU)",
        ).pack(anchor="w")
        ram = total_ram_gb()
        if ram is not None and ram < 12:
            ttk.Label(
                f, wraplength=800, justify="left", foreground="#a33",
                text=(f"This computer has {ram} GB of RAM. Voice cloning needs a lot of memory and is very slow "
                      "without a graphics card, so it is not recommended here. The generic voice works well."),
            ).pack(anchor="w", pady=(2, 0))
        row = ttk.Frame(f)
        row.pack(anchor="w", pady=8)
        self.setup_btn = ttk.Button(row, text="Set up now", command=self._start_setup)
        self.setup_btn.pack(side="left")
        ttk.Button(row, text="Open my data folder", command=lambda: core.open_path(core.data_dir())).pack(side="left", padx=8)
        self.setup_log = self._log_box(f)

    def _refresh_state(self) -> None:
        ready = core.is_setup_done()
        self.setup_status.configure(text="Ready. You can build a twin." if ready else "Setup needed before first use.")
        self.setup_btn.configure(text="Repair / update setup" if ready else "Set up now")
        self._refresh_twins()

    def _start_setup(self) -> None:
        voice = self.voice_var.get()
        if voice and not messagebox.askyesno("Voice cloning licence", XTTS_TERMS):
            voice = False
            self.voice_var.set(False)
        self.setup_btn.configure(state="disabled")
        log = self._logger(self.setup_log)

        def work() -> None:
            try:
                have_ollama = core.full_setup(log, voice)
                if not have_ollama:
                    log("The language-model runner (Ollama) is not installed.")
                    if self._ask("Install Ollama?",
                                 "Ollama runs the language model on your computer. Install it now? "
                                 "Its own installer window will open; click through it."):
                        if core.install_ollama(log):
                            core.run_install(log, voice, only_llm=True)
                        else:
                            log("Ollama was not detected. Install it from https://ollama.com, then press "
                                "'Repair / update setup'.")
                else:
                    core.run_install(log, voice, only_llm=True)
                log("Setup finished.")
            finally:
                self._post(self.setup_btn.configure, {"state": "normal"})
                self._post(self._refresh_state)

        self._run_bg(work, self.setup_log)

    def _ask(self, title: str, text: str) -> bool:
        """Ask a yes/no question from a worker thread."""
        answer: queue.Queue = queue.Queue()
        self._post(lambda: answer.put(messagebox.askyesno(title, text)))
        return answer.get()

    # ------------------------------------------------------------- tab 2: build
    def _build_build(self) -> None:
        f = self.build_tab
        grid = ttk.Frame(f)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        self.folder_var = tk.StringVar()
        ttk.Label(grid, text="Folder with their photos, videos, chats, documents:").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Entry(grid, textvariable=self.folder_var, state="readonly").grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(grid, text="Choose folder...", command=self._choose_folder).grid(row=1, column=2, padx=(6, 0))
        self.scan_label = ttk.Label(grid, wraplength=780, justify="left")
        self.scan_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 8))

        ttk.Label(grid, text="Which name in the chats is the person?").grid(row=3, column=0, sticky="w")
        self.speaker_var = tk.StringVar()
        self.speaker_box = ttk.Combobox(grid, textvariable=self.speaker_var)
        self.speaker_box.grid(row=3, column=1, columnspan=2, sticky="ew", pady=2)
        self.speaker_box.bind("<<ComboboxSelected>>", self._speaker_picked)

        ttk.Label(grid, text="Name for this twin:").grid(row=4, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.name_var).grid(row=4, column=1, columnspan=2, sticky="ew", pady=2)

        ttk.Label(grid, text="A clear photo of their face (recommended):").grid(row=5, column=0, sticky="w")
        self.photo_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.photo_var, state="readonly").grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Button(grid, text="Choose photo...", command=self._choose_photo).grid(row=5, column=2, padx=(6, 0))
        ttk.Label(grid, wraplength=780, justify="left", foreground="#555",
                  text="Front-facing, mouth closed, eyes open, good light. Without one, a photo from the folder is guessed, "
                       "and it may be the wrong person.").grid(row=6, column=0, columnspan=3, sticky="w")

        ttk.Label(grid, text="Who will be talking to them?").grid(row=9, column=0, sticky="w", pady=(8, 0))
        self.listener_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.listener_var).grid(row=9, column=1, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Label(grid, foreground="#555",
                  text='For example "his daughter Meera" or "her grandson Rahul". It changes how they answer you.'
                  ).grid(row=10, column=1, columnspan=2, sticky="w")

        ttk.Label(grid, text="Language they speak:").grid(row=11, column=0, sticky="w", pady=(10, 0))
        self.language_var = tk.StringVar(value=DEFAULT_LANGUAGE)
        language_box = ttk.Combobox(grid, textvariable=self.language_var, values=list(LANGUAGES), state="readonly")
        language_box.grid(row=11, column=1, sticky="w", pady=(10, 2))
        language_box.bind("<<ComboboxSelected>>", self._language_picked)
        ttk.Label(grid, foreground="#555",
                  text="Also used for transcribing recordings and, if the cloned voice is used, for how it speaks."
                  ).grid(row=12, column=1, columnspan=2, sticky="w")
        self.language_tip = ttk.Label(grid, foreground="#a60", wraplength=780, justify="left")
        self.language_tip.grid(row=13, column=1, columnspan=2, sticky="w", pady=(4, 0))
        self.language_model_row = ttk.Frame(grid)
        self.language_model_row.grid(row=14, column=1, columnspan=2, sticky="w")
        self.use_language_model_var = tk.BooleanVar(value=False)
        self.language_model_check = ttk.Checkbutton(
            self.language_model_row, variable=self.use_language_model_var, text="Use it for this twin")
        self.language_model_check.pack(side="left")
        self.language_download_btn = ttk.Button(
            self.language_model_row, text="Download now", command=self._download_language_model)
        self.language_download_btn.pack(side="left", padx=8)
        self.language_model_status = ttk.Label(self.language_model_row, foreground="#555")
        self.language_model_status.pack(side="left")
        self.language_model_row.grid_remove()
        self._suggested_model = ""
        self._suggested_model_size = 0.0

        ttk.Label(grid, text="Notes about them:").grid(row=15, column=0, sticky="nw", pady=(6, 0))
        self.notes_box = scrolledtext.ScrolledText(grid, height=5, wrap="word")
        self.notes_box.grid(row=15, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(grid, foreground="#555", wraplength=780, justify="left",
                  text=("Chats mostly contain small talk. Facts, stories and habits only exist if you write them here: "
                        "what you called each other, favourite sayings, where they grew up, what they cared about, "
                        "how they greeted you.")).grid(row=16, column=1, columnspan=2, sticky="w")

        self.media_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, variable=self.media_var,
                        text="Also use audio and video recordings (only if just this person speaks in them)"
                        ).grid(row=17, column=0, columnspan=3, sticky="w", pady=(8, 0))
        minutes = ttk.Frame(grid)
        minutes.grid(row=18, column=0, columnspan=3, sticky="w")
        ttk.Label(minutes, text="Listen to at most").pack(side="left")
        self.minutes_var = tk.IntVar(value=90)
        ttk.Spinbox(minutes, from_=5, to=600, increment=5, width=5, textvariable=self.minutes_var).pack(side="left", padx=4)
        ttk.Label(minutes, text="minutes of recordings (short voice notes first). Longer takes longer.").pack(side="left")

        self.consent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, variable=self.consent_var,
                        text="I confirm I am allowed to build this twin (see below).").pack(anchor="w", pady=(10, 0))
        ttk.Label(f, text=CONSENT, wraplength=800, justify="left", foreground="#555").pack(anchor="w")
        self.build_btn = ttk.Button(f, text="Build the twin", command=self._start_build)
        self.build_btn.pack(anchor="w", pady=8)
        self.build_log = self._log_box(f)

    def _language_picked(self, _event=None) -> None:
        code = code_for(self.language_var.get())
        tip = recommend_llm_for_language(code, total_ram_gb(), has_nvidia_gpu(), is_apple_silicon()) if code else None
        self.use_language_model_var.set(False)
        self.language_model_status.configure(text="")
        if not tip:
            self.language_tip.configure(text="")
            self.language_model_row.grid_remove()
            self._suggested_model = ""
            self._suggested_model_size = 0.0
            return
        fits = "" if tip["fits"] else " This computer may be too slow for it, though."
        self.language_tip.configure(text=f"{tip['reason']}{fits}")
        self.language_model_row.grid()
        self._suggested_model = tip["model"]
        self._suggested_model_size = tip["size_gb"]
        self._refresh_language_model_status()

    def _refresh_language_model_status(self) -> None:
        model = self._suggested_model
        if not model:
            return
        try:
            have = self._model_is_installed(model)
        except Exception:
            have = None
        if have:
            self.language_model_status.configure(text=f"{model} is already installed.")
            self.use_language_model_var.set(True)
        elif have is False:
            self.language_model_status.configure(
                text=f"Not installed yet ({self._suggested_model_size:.1f} GB download).")
        else:
            self.language_model_status.configure(text="")

    def _model_is_installed(self, model: str) -> bool | None:
        """True/False once Ollama answers, None if it cannot be reached (e.g. not set up yet)."""
        import json
        import urllib.request

        from twin.ollama import OLLAMA_URL

        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
                installed = {m.get("name", "") for m in json.loads(resp.read()).get("models", [])}
        except Exception:
            return None
        wanted = model if ":" in model else f"{model}:latest"
        return wanted in installed

    def _download_language_model(self) -> None:
        if not core.is_setup_done():
            messagebox.showinfo("Set up first", "Please finish the one-time setup on the first tab.")
            self.tabs.select(self.setup_tab)
            return
        model, size = self._suggested_model, self._suggested_model_size
        if not model:
            return
        if not messagebox.askyesno(
            "Download language model",
            f"Download {model}? This is about {size:.1f} GB and needs the internet. "
            "It only has to be done once."
        ):
            return
        self.language_download_btn.configure(state="disabled")
        self.language_model_status.configure(text="Downloading...")
        log = self._logger(self.build_log)

        def work() -> None:
            log(f"--- downloading {model} ---")
            ok = core.download_language_model(model, log)
            self._post(self.language_download_btn.configure, {"state": "normal"})
            if ok:
                self._post(self.language_model_status.configure, {"text": f"{model} is ready."})
                self._post(self.use_language_model_var.set, True)
                log(f"{model} downloaded and ready.")
            else:
                self._post(self.language_model_status.configure, {"text": "Download did not finish."})
                self._post(messagebox.showerror, "Download failed",
                           f"Could not download {model}. Check the internet connection and the log below, then try again.")

        self._run_bg(work, self.build_log)

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose the folder with their material")
        if not folder:
            return
        self.folder_var.set(folder)
        self.scan_label.configure(text="Looking inside the folder...")

        def work() -> None:
            result = scan_folder(Path(folder))
            self._post(self._scan_done, result)

        self._run_bg(work)

    def _scan_done(self, result) -> None:
        self.scan_speakers = result.speakers
        self.scan_label.configure(text="Found: " + result.summary())
        self.speaker_box.configure(values=[f"{n}  ({c} messages)" for n, c in result.speakers])
        if result.speakers:
            self.speaker_box.current(0)
            self._speaker_picked()

    def _speaker_name(self) -> str:
        return self.speaker_var.get().split("  (")[0].strip()

    def _speaker_picked(self, _event=None) -> None:
        name = self._speaker_name()
        self.speaker_var.set(name)
        if not self.name_var.get():
            self.name_var.set(name)

    def _choose_photo(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a clear photo of their face",
            filetypes=[("Photos", "*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp"), ("All files", "*.*")])
        if path:
            self.photo_var.set(path)

    def _start_build(self) -> None:
        if not core.is_setup_done():
            messagebox.showinfo("Set up first", "Please finish the one-time setup on the first tab.")
            self.tabs.select(self.setup_tab)
            return
        folder, name = self.folder_var.get(), self.name_var.get().strip()
        speaker = self._speaker_name() or name
        if not folder or not name:
            messagebox.showinfo("Missing information", "Choose a folder and give the twin a name.")
            return
        if not self.consent_var.get():
            messagebox.showinfo("Permission needed", "Please confirm the permission statement first.")
            return
        notes = self.notes_box.get("1.0", "end").strip()
        about_file = ""
        if notes:
            notes_path = core.data_dir() / "_about_input.txt"
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            notes_path.write_text(notes, encoding="utf-8")
            about_file = str(notes_path)
        listener = self.listener_var.get().strip()
        language = code_for(self.language_var.get())
        llm_model = self._suggested_model if self.use_language_model_var.get() else ""
        cmd = core.ingest_command(name, folder, speaker, self.photo_var.get(), self.media_var.get(),
                                  int(self.minutes_var.get()), about_file, listener, language, llm_model)
        self.build_btn.configure(state="disabled")
        log = self._logger(self.build_log)

        def work() -> None:
            try:
                log(f"Building '{name}'. This can take a while for large folders.")
                code = core.stream(cmd, log, cwd=core.app_dir())
                if code == 0:
                    log("Done. Open the Talk tab.")
                    self._post(self.talk_listener_var.set, listener)
                    self._post(self.talk_language_var.set, self.language_var.get())
                    self._post(self._refresh_twins, name)
                    self._post(self.tabs.select, self.talk_tab)
                else:
                    log(f"The build stopped with an error (code {code}).")
            finally:
                self._post(self.build_btn.configure, {"state": "normal"})

        self._run_bg(work, self.build_log)

    # ------------------------------------------------------------- tab 3: talk
    def _build_talk(self) -> None:
        f = self.talk_tab
        top = ttk.Frame(f)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Twin:").grid(row=0, column=0, sticky="w")
        self.twin_var = tk.StringVar()
        self.twin_box = ttk.Combobox(top, textvariable=self.twin_var, state="readonly")
        self.twin_box.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(top, text="Delete...", command=self._delete_twin).grid(row=0, column=2)

        ttk.Label(top, text="Show it on:").grid(row=1, column=0, sticky="w")
        self.layout_var = tk.StringVar(value=list(core.LAYOUTS)[0])
        ttk.Combobox(top, textvariable=self.layout_var, values=list(core.LAYOUTS), state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(top, text="Voice:").grid(row=2, column=0, sticky="w")
        self.voice_choice = tk.StringVar(value=list(VOICE_CHOICES)[0])
        ttk.Combobox(top, textvariable=self.voice_choice, values=list(VOICE_CHOICES), state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(top, text="Language:").grid(row=3, column=0, sticky="w")
        self.talk_language_var = tk.StringVar(value="(use the twin's saved language)")
        self.talk_language_box = ttk.Combobox(
            top, textvariable=self.talk_language_var,
            values=["(use the twin's saved language)"] + list(LANGUAGES), state="readonly")
        self.talk_language_box.grid(row=3, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(top, text="Face photo:").grid(row=4, column=0, sticky="w")
        self.talk_photo_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.talk_photo_var, state="readonly").grid(row=4, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(top, text="Choose...", command=self._choose_talk_photo).grid(row=4, column=2)
        ttk.Label(top, foreground="#555", text="(optional: overrides the photo chosen when building)").grid(
            row=5, column=1, sticky="w", padx=6)

        ttk.Label(top, text="You are their:").grid(row=6, column=0, sticky="w")
        self.talk_listener_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.talk_listener_var).grid(row=6, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(top, foreground="#555", text='e.g. "his daughter Meera"').grid(row=6, column=2, sticky="w")

        opts = ttk.Frame(top)
        opts.grid(row=7, column=0, columnspan=3, sticky="w", pady=4)
        self.mic_var, self.mirror_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, variable=self.mic_var, text="Talk with the microphone").pack(side="left")
        ttk.Checkbutton(opts, variable=self.mirror_var, text="Mirror the picture (rear projection)").pack(side="left", padx=10)
        self.show_mem_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, variable=self.show_mem_var, text="Show what it remembered").pack(side="left", padx=10)
        ttk.Label(opts, text="Screen #").pack(side="left")
        self.screen_var = tk.IntVar(value=0)
        ttk.Spinbox(opts, from_=0, to=4, width=3, textvariable=self.screen_var).pack(side="left", padx=4)

        btns = ttk.Frame(f)
        btns.pack(anchor="w", pady=6)
        self.start_btn = ttk.Button(btns, text="Start", command=self._start_chat)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.session.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Test sound", command=self._test_sound).pack(side="left", padx=6)
        ttk.Button(btns, text="About this twin", command=self._inspect_twin).pack(side="left")
        ttk.Label(btns, text="Close the picture window with Esc.").pack(side="left", padx=8)

        self.transcript = self._log_box(f)
        entry_row = ttk.Frame(f)
        entry_row.pack(fill="x", pady=6)
        self.say_var = tk.StringVar()
        entry = ttk.Entry(entry_row, textvariable=self.say_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._send())
        ttk.Button(entry_row, text="Send", command=self._send).pack(side="left", padx=6)

    def _choose_talk_photo(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a clear front-facing photo",
            filetypes=[("Photos", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")])
        if path:
            self.talk_photo_var.set(path)

    def _inspect_twin(self) -> None:
        name = self.twin_var.get()
        if not name:
            return
        log = self._logger(self.transcript)
        log("--- what this twin has learned ---")
        self._run_bg(lambda: core.stream(core.inspect_command(name), log, cwd=core.app_dir()), self.transcript)

    def _test_sound(self) -> None:
        if not core.is_setup_done():
            messagebox.showinfo("Set up first", "Please finish the one-time setup first.")
            return
        log = self._logger(self.transcript)
        log("--- sound test ---")
        self._run_bg(lambda: log(f"[sound test finished, code {core.stream(core.soundtest_command(), log, cwd=core.app_dir())}]"),
                     self.transcript)

    def _refresh_twins(self, select: str | None = None) -> None:
        names = core.list_twins()
        self.twin_box.configure(values=names)
        if select and select in names:
            self.twin_var.set(select)
        elif names and not self.twin_var.get():
            self.twin_var.set(names[0])

    def _delete_twin(self) -> None:
        name = self.twin_var.get()
        if not name:
            return
        if messagebox.askyesno("Delete twin", f"Permanently delete '{name}' and everything built from their material?"):
            self.session.stop()
            shutil.rmtree(core.data_dir() / slugify(name), ignore_errors=True)
            self.twin_var.set("")
            self._refresh_twins()

    def _start_chat(self) -> None:
        name = self.twin_var.get()
        if not name:
            messagebox.showinfo("Choose a twin", "Build a twin first, then pick it here.")
            return
        if self.session.running:
            return
        voice, speak = VOICE_CHOICES[self.voice_choice.get()]
        cmd = core.chat_command(name, self.layout_var.get(), self.mirror_var.get(), int(self.screen_var.get()),
                                self.mic_var.get(), voice, speak, self.talk_photo_var.get(),
                                self.talk_listener_var.get().strip(),
                                code_for(self.talk_language_var.get()), self.show_mem_var.get())
        log = self._logger(self.transcript)
        self.start_btn.configure(state="disabled")

        def work() -> None:
            if not core.ensure_ollama_running(log):
                self._post(messagebox.showerror, "Language model not running",
                           "Ollama is not installed or could not start. Use the Set up tab to install it.")
                self._post(self.start_btn.configure, {"state": "normal"})
                return
            log("Starting... the first reply can take a minute while models load.")

            def exited(code: int) -> None:
                self._post(self.start_btn.configure, {"state": "normal"})
                self._post(self.stop_btn.configure, {"state": "disabled"})
                log(f"[session ended, code {code}]")

            self.session.start(cmd, log, exited)
            self._post(self.stop_btn.configure, {"state": "normal"})

        self._run_bg(work, self.transcript)

    def _send(self) -> None:
        text = self.say_var.get().strip()
        if not text:
            return
        if self.session.send(text):
            self._append(self.transcript, f"You: {text}")
            self.say_var.set("")
        else:
            messagebox.showinfo("Not running", "Press Start first.")
