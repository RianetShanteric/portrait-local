"""Small desktop front-end for the local portrait pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageOps


BG = "#0b0d12"
PANEL = "#131720"
PANEL_ALT = "#191e29"
TEXT = "#f4f6fb"
MUTED = "#9299a9"
ACCENT = "#9d7cff"
ACCENT_HOVER = "#b097ff"
MINT = "#67e8c2"
DANGER = "#ff6b81"

PROFILES = {
    "Естественно": "natural",
    "Баланс": "balanced",
    "Сильно": "strong",
}

SCENE_LABELS = {
    "portrait": "Портрет", "group": "Группа", "landscape": "Пейзаж",
    "city": "Город", "architecture": "Архитектура", "indoor": "В помещении",
    "night": "Ночной кадр", "food": "Еда", "animal": "Животное",
    "object": "Предмет", "general": "Обычный кадр",
}


class PortraitApp(ctk.CTk):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(fg_color=BG)
        self.root_dir = root_dir
        self.input_dir = root_dir / "data" / "input"
        self.output_dir = root_dir / "data" / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources: list[Path] = []
        self.selected: Path | None = None
        self.current_result: Path | None = None
        self.preview_image: ctk.CTkImage | None = None
        self.running = False

        self.title("Portrait Local")
        self.geometry("1080x720")
        self.minsize(900, 620)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_queue()
        self._build_preview()
        self.refresh_files()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent", height=96)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=30, pady=(24, 14))
        header.grid_columnconfigure(0, weight=1)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            brand,
            text="PORTRAIT / LOCAL",
            text_color=ACCENT,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="Тихая ретушь без чужого лица",
            text_color=TEXT,
            font=("Segoe UI Variable Display", 29, "bold"),
        ).pack(anchor="w", pady=(3, 0))

        badges = ctk.CTkFrame(header, fg_color="transparent")
        badges.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(
            badges,
            text="  ●  ЛОКАЛЬНО  ",
            fg_color="#14362f",
            text_color=MINT,
            corner_radius=10,
            height=32,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=5)
        ctk.CTkLabel(
            badges,
            text="  RTX 5070  ",
            fg_color="#262039",
            text_color="#cdbfff",
            corner_radius=10,
            height=32,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=5)

    def _build_queue(self) -> None:
        left = ctk.CTkFrame(self, width=350, fg_color=PANEL, corner_radius=22)
        left.grid(row=1, column=0, sticky="nsew", padx=(30, 12), pady=(0, 28))
        left.grid_propagate(False)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="Фотографии",
            text_color=TEXT,
            font=("Segoe UI Variable Display", 20, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 2))
        self.count_label = ctk.CTkLabel(
            left,
            text="INPUT пуст",
            text_color=MUTED,
            font=("Segoe UI", 12),
        )
        self.count_label.grid(row=1, column=0, sticky="w", padx=22, pady=(0, 12))

        self.file_list = ctk.CTkScrollableFrame(left, fg_color="transparent", corner_radius=0)
        self.file_list.grid(row=2, column=0, sticky="nsew", padx=12)
        self.file_list.grid_columnconfigure(0, weight=1)

        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", padx=18, pady=(12, 8))
        controls.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            controls,
            text="＋ Добавить",
            command=self.add_files,
            height=40,
            corner_radius=12,
            fg_color=PANEL_ALT,
            hover_color="#252c3a",
            border_width=1,
            border_color="#2b3241",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(
            controls,
            text="Открыть INPUT",
            command=lambda: os.startfile(self.input_dir),
            height=40,
            corner_radius=12,
            fg_color=PANEL_ALT,
            hover_color="#252c3a",
            border_width=1,
            border_color="#2b3241",
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.profile = ctk.StringVar(value="Баланс")
        ctk.CTkLabel(
            left,
            text="Сила обработки",
            text_color=MUTED,
            font=("Segoe UI", 11, "bold"),
        ).grid(row=4, column=0, sticky="w", padx=22, pady=(10, 5))
        self.profile_control = ctk.CTkSegmentedButton(
            left,
            values=list(PROFILES),
            variable=self.profile,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_color=PANEL_ALT,
            unselected_hover_color="#252c3a",
            corner_radius=11,
            height=36,
            font=("Segoe UI", 11, "bold"),
        )
        self.profile_control.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 14))

        self.run_button = ctk.CTkButton(
            left,
            text="ОБРАБОТАТЬ",
            command=self.start_processing,
            height=56,
            corner_radius=16,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#111018",
            font=("Segoe UI Variable Display", 15, "bold"),
        )
        self.run_button.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.progress = ctk.CTkProgressBar(
            left, mode="indeterminate", height=4, progress_color=MINT, fg_color="#242936"
        )
        self.progress.grid(row=7, column=0, sticky="ew", padx=22, pady=(0, 7))
        self.progress.set(0)
        self.status_label = ctk.CTkLabel(
            left,
            text="Готова к работе",
            text_color=MUTED,
            font=("Segoe UI", 11),
        )
        self.status_label.grid(row=8, column=0, padx=20, pady=(0, 18))

    def _build_preview(self) -> None:
        right = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=22)
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 30), pady=(0, 28))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(right, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 10))
        top.grid_columnconfigure(0, weight=1)
        self.preview_title = ctk.CTkLabel(
            top,
            text="Предпросмотр",
            text_color=TEXT,
            font=("Segoe UI Variable Display", 18, "bold"),
        )
        self.preview_title.grid(row=0, column=0, sticky="w")
        self.preview_mode = ctk.StringVar(value="Исходник")
        self.preview_switch = ctk.CTkSegmentedButton(
            top,
            values=["Исходник", "Результат"],
            variable=self.preview_mode,
            command=lambda _value: self.show_preview(),
            width=210,
            selected_color="#353049",
            selected_hover_color="#40395a",
            unselected_color="#1a1f2a",
            unselected_hover_color="#252b38",
        )
        self.preview_switch.grid(row=0, column=1, sticky="e")

        self.preview_box = ctk.CTkFrame(right, fg_color="#080a0e", corner_radius=17)
        self.preview_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.preview_box.grid_rowconfigure(0, weight=1)
        self.preview_box.grid_columnconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(
            self.preview_box,
            text="＋\n\nДобавь фотографии слева",
            text_color="#646b7a",
            font=("Segoe UI", 14),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        footer = ctk.CTkFrame(right, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 18))
        footer.grid_columnconfigure(0, weight=1)
        self.detail_label = ctk.CTkLabel(
            footer,
            text="Фото остаются на этом компьютере",
            text_color=MUTED,
            font=("Segoe UI", 11),
        )
        self.detail_label.grid(row=0, column=0, sticky="w")
        self.open_output_button = ctk.CTkButton(
            footer,
            text="Открыть результат  ↗",
            command=self.open_latest_output,
            state="disabled",
            width=170,
            height=36,
            corner_radius=11,
            fg_color="#17372f",
            hover_color="#205044",
            text_color=MINT,
        )
        self.open_output_button.grid(row=0, column=1, sticky="e")

    def refresh_files(self, select: Path | None = None) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.sources = sorted(
            (path for path in self.input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions),
            key=lambda path: path.name.lower(),
        )
        for widget in self.file_list.winfo_children():
            widget.destroy()
        for index, source in enumerate(self.sources):
            row = ctk.CTkFrame(self.file_list, fg_color=PANEL_ALT, corner_radius=12, height=48)
            row.grid(row=index, column=0, sticky="ew", pady=4)
            row.grid_columnconfigure(0, weight=1)
            name = ctk.CTkButton(
                row,
                text=source.name,
                command=lambda item=source: self.select_source(item),
                anchor="w",
                fg_color="transparent",
                hover_color="#252b38",
                text_color=TEXT,
                corner_radius=10,
                height=44,
                font=("Segoe UI", 12),
            )
            name.grid(row=0, column=0, sticky="ew", padx=(3, 0), pady=2)
            remove = ctk.CTkButton(
                row,
                text="×",
                command=lambda item=source: self.remove_source(item),
                width=34,
                height=34,
                fg_color="transparent",
                hover_color="#41212a",
                text_color=MUTED,
                corner_radius=9,
                font=("Segoe UI", 18),
            )
            remove.grid(row=0, column=1, padx=5)
        count = len(self.sources)
        self.count_label.configure(text="INPUT пуст" if not count else f"В очереди: {count}")
        self.run_button.configure(text="ОБРАБОТАТЬ" if not count else f"ОБРАБОТАТЬ  ·  {count}")
        if select and select in self.sources:
            self.select_source(select)
        elif self.sources and self.selected not in self.sources:
            self.select_source(self.sources[0])
        elif not self.sources:
            self.selected = None
            self.preview_image = None
            self.preview_label.configure(image=None, text="＋\n\nДобавь фотографии слева")

    def select_source(self, source: Path) -> None:
        self.selected = source
        self.preview_mode.set("Исходник")
        self.show_preview()

    def add_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Добавить портреты",
            filetypes=[("Фотографии", "*.jpg *.jpeg *.png *.webp"), ("Все файлы", "*.*")],
        )
        if not selected:
            return
        last_added: Path | None = None
        for raw in selected:
            source = Path(raw)
            destination = self.input_dir / source.name
            if source.resolve() == destination.resolve():
                last_added = destination
                continue
            counter = 2
            while destination.exists():
                destination = self.input_dir / f"{source.stem}_{counter}{source.suffix}"
                counter += 1
            shutil.copy2(source, destination)
            last_added = destination
        self.refresh_files(last_added)

    def remove_source(self, source: Path) -> None:
        if self.running:
            return
        try:
            source.unlink()
        except OSError as error:
            messagebox.showerror("Не получилось удалить", str(error), parent=self)
            return
        self.refresh_files()

    def show_preview(self) -> None:
        path = self.selected
        if self.preview_mode.get() == "Результат" and self.current_result:
            path = self.current_result
        if not path or not path.exists():
            self.preview_label.configure(image=None, text="Результат появится после обработки")
            return
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                max_width = max(360, self.preview_box.winfo_width() - 40)
                max_height = max(360, self.preview_box.winfo_height() - 40)
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                self.preview_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.preview_label.configure(image=self.preview_image, text="")
            self.detail_label.configure(text=f"{path.name}  ·  {image.width} × {image.height} preview")
        except Exception as error:
            self.preview_label.configure(image=None, text=f"Не удалось открыть превью\n{error}")

    def start_processing(self) -> None:
        if self.running:
            return
        self.refresh_files(self.selected)
        if not self.sources:
            messagebox.showinfo("INPUT пуст", "Добавь хотя бы одну фотографию.", parent=self)
            return
        self.running = True
        self.run_button.configure(state="disabled", text="ОБРАБАТЫВАЮ…")
        self.profile_control.configure(state="disabled")
        self.open_output_button.configure(state="disabled")
        self.status_label.configure(text="RTX считает локально · окно можно оставить в фоне", text_color=MINT)
        self.progress.start()
        profile = PROFILES[self.profile.get()]
        thread = threading.Thread(target=self._process_worker, args=(profile,), daemon=True)
        thread.start()

    def _process_worker(self, profile: str) -> None:
        command = [
            sys.executable,
            "-u",
            "-m",
            "app.cli",
            "--root",
            str(self.root_dir),
            "--profile",
            profile,
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            result = subprocess.run(
                command,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            self.after(0, self._processing_finished, result.returncode, result.stdout, result.stderr)
        except Exception as error:
            self.after(0, self._processing_finished, 1, "", f"{error}\n{traceback.format_exc()}")

    def _processing_finished(self, return_code: int, stdout: str, stderr: str) -> None:
        self.running = False
        self.progress.stop()
        self.progress.set(0)
        self.profile_control.configure(state="normal")
        self.run_button.configure(state="normal")
        self.run_button.configure(text=f"ОБРАБОТАТЬ  ·  {len(self.sources)}")
        if return_code != 0:
            error_log = self.root_dir / "runtime" / "logs" / "ui-last-error.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            error_log.write_text(stdout + "\n" + stderr, encoding="utf-8")
            self.status_label.configure(text="Что-то сломалось · детали сохранены в runtime/logs", text_color=DANGER)
            messagebox.showerror(
                "Обработка не завершилась",
                "Подробности сохранены в runtime\\logs\\ui-last-error.log.",
                parent=self,
            )
            return

        folders = sorted((path for path in self.output_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
        latest = folders[-1] if folders else None
        self.latest_output = latest
        self.current_result = None
        if latest and self.selected:
            for recipe_path in latest.glob("*.recipe.json"):
                try:
                    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                    if recipe.get("input", {}).get("name") != self.selected.name:
                        continue
                    output_name = recipe.get("output", {}).get("name")
                    candidate = latest / output_name if output_name else None
                    if candidate and candidate.is_file():
                        self.current_result = candidate
                        break
                except (OSError, ValueError, TypeError):
                    continue
        self.status_label.configure(text=f"Готово · {len(self.sources)} фото", text_color=MINT)
        self.open_output_button.configure(state="normal")
        if self.current_result:
            recipe_path = self.current_result.with_suffix(".recipe.json")
            try:
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                metrics = recipe.get("metrics", {})
                scene = SCENE_LABELS.get(metrics.get("scene"), "Кадр")
                confidence = round(float(metrics.get("scene_confidence", 0.0)) * 100)
                self.status_label.configure(
                    text=f"Готово · {len(self.sources)} фото · {scene} {confidence}%",
                    text_color=MINT,
                )
            except (OSError, ValueError, TypeError):
                pass
            self.preview_mode.set("Результат")
            self.show_preview()

    def open_latest_output(self) -> None:
        latest = getattr(self, "latest_output", None)
        os.startfile(latest if latest and latest.exists() else self.output_dir)


def install_exception_log(root: Path) -> None:
    def handler(exc_type, exc_value, exc_traceback):
        log = root / "runtime" / "logs" / "ui-crash.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)), encoding="utf-8")
        messagebox.showerror("Portrait Local", f"Ошибка интерфейса сохранена в:\n{log}")

    sys.excepthook = handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    install_exception_log(root)
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = PortraitApp(root)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
