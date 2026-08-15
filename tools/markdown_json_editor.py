"""Edit a Markdown file using guidance from a JSON file.

Usage:
    python tools/markdown_json_editor.py edits.json draft.md

The JSON can be a list of edit objects, a dictionary containing a common list key
such as "edits" or "suggestions", or a simple object whose values should be shown
as editable suggestions.

Entries whose find text is no longer present in the Markdown are hidden from the
pending list by default. Successfully applied entries are also hidden for the
current editing session.

Commonly recognized fields:
    title, label, name, section, heading, id
    find, search, match, original, before, old_text
    replacement, replace_with, suggestion, suggested_text, after, new_text, text
    line, line_number, start_line

Example JSON:
    [
      {
        "title": "Tighten the intro",
        "find": "This post is going to talk about dataset curation.",
        "replacement": "Dataset curation looks boring until it becomes the bottleneck."
      },
      {
        "section": "Conclusion",
        "line": 120,
        "suggestion": "Add a short paragraph tying this back to production use."
      }
    ]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ENTRY_LIST_KEYS = (
    "edits",
    "suggestions",
    "changes",
    "items",
    "annotations",
    "comments",
    "reviews",
    "recommendations",
)

TITLE_KEYS = (
    "title",
    "label",
    "name",
    "section",
    "heading",
    "id",
    "summary",
)

FIND_KEYS = (
    "find",
    "search",
    "match",
    "target",
    "from",
    "selection",
    "original",
    "before",
    "old",
    "oldtext",
    "old_text",
    "source",
    "current",
)

REPLACEMENT_KEYS = (
    "replacement",
    "replace",
    "replacewith",
    "replace_with",
    "to",
    "suggestion",
    "suggested",
    "suggestedtext",
    "suggested_text",
    "new",
    "newtext",
    "new_text",
    "after",
    "revised",
    "revision",
    "text",
    "content",
    "value",
)

LINE_KEYS = (
    "line",
    "lineno",
    "line_no",
    "linenumber",
    "line_number",
    "startline",
    "start_line",
)

DETAIL_KEYS = (
    "comment",
    "issue",
    "reason",
    "description",
    "note",
    "notes",
    "rationale",
)


@dataclass(frozen=True)
class JsonEntry:
    path: str
    title: str
    data: Any
    find_text: str
    replacement_text: str
    line_number: int | None


def normalize_key(key: str) -> str:
    return "".join(char for char in key.lower() if char.isalnum())


def key_matches(key: str, candidates: Iterable[str]) -> bool:
    normalized = normalize_key(key)
    return normalized in {normalize_key(candidate) for candidate in candidates}


def preview(value: Any, limit: int = 90) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2)


def first_text_for_keys(value: Any, keys: Iterable[str]) -> str:
    if not isinstance(value, dict):
        return value_as_text(value) if isinstance(value, str) else ""

    for key, item in value.items():
        if key_matches(key, keys):
            return value_as_text(item)

    for nested_key in ("edit", "change", "patch", "suggestion", "replacement"):
        for key, item in value.items():
            if normalize_key(key) == normalize_key(nested_key) and isinstance(item, dict):
                found = first_text_for_keys(item, keys)
                if found:
                    return found

    return ""


def first_int_for_keys(value: Any, keys: Iterable[str]) -> int | None:
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if not key_matches(key, keys):
            continue
        if isinstance(item, int):
            return item
        if isinstance(item, str) and item.strip().isdigit():
            return int(item.strip())
    return None


def title_for_entry(path: str, value: Any) -> str:
    if isinstance(value, dict):
        for key in TITLE_KEYS:
            for item_key, item in value.items():
                if key_matches(item_key, (key,)) and value_as_text(item).strip():
                    return value_as_text(item).strip()
        for key in DETAIL_KEYS:
            for item_key, item in value.items():
                if key_matches(item_key, (key,)) and value_as_text(item).strip():
                    return preview(item)
        find_text = first_text_for_keys(value, FIND_KEYS)
        replacement_text = first_text_for_keys(value, REPLACEMENT_KEYS)
        if find_text and replacement_text:
            return f"Replace: {preview(find_text, 70)}"
        if replacement_text:
            return preview(replacement_text, 70)

    if isinstance(value, str):
        return preview(value, 70)

    return path


def looks_like_edit(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, dict):
        return False
    known_groups = (TITLE_KEYS, FIND_KEYS, REPLACEMENT_KEYS, LINE_KEYS, DETAIL_KEYS)
    return any(key_matches(key, group) for key in value for group in known_groups)


def make_entry(path: str, value: Any) -> JsonEntry:
    return JsonEntry(
        path=path,
        title=title_for_entry(path, value),
        data=value,
        find_text=first_text_for_keys(value, FIND_KEYS),
        replacement_text=first_text_for_keys(value, REPLACEMENT_KEYS),
        line_number=first_int_for_keys(value, LINE_KEYS),
    )


def entries_from_list(path: str, values: list[Any]) -> list[JsonEntry]:
    entries: list[JsonEntry] = []
    for index, item in enumerate(values, start=1):
        item_path = f"{path}[{index - 1}]"
        if looks_like_edit(item):
            entries.append(make_entry(item_path, item))
        elif isinstance(item, (dict, list)):
            entries.extend(build_entries(item, item_path))
        else:
            entries.append(make_entry(item_path, item))
    return entries


def build_entries(value: Any, path: str = "$") -> list[JsonEntry]:
    if isinstance(value, list):
        return entries_from_list(path, value)

    if isinstance(value, dict):
        for entry_key in ENTRY_LIST_KEYS:
            for key, item in value.items():
                if key_matches(key, (entry_key,)) and isinstance(item, list):
                    return entries_from_list(f"{path}.{key}", item)

        if looks_like_edit(value):
            return [make_entry(path, value)]

        entries: list[JsonEntry] = []
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if looks_like_edit(item):
                entries.append(make_entry(item_path, item))
            elif isinstance(item, (dict, list)):
                entries.extend(build_entries(item, item_path))
            else:
                entries.append(make_entry(item_path, {"title": key, "value": item}))
        return entries or [make_entry(path, value)]

    return [make_entry(path, value)]


def markdown_position_for_entry(markdown: str, entry: JsonEntry) -> tuple[int, int]:
    if entry.line_number is not None:
        return (entry.line_number, 0)
    if entry.find_text:
        position = markdown.find(entry.find_text)
        if position != -1:
            return (markdown.count("\n", 0, position) + 1, position)
    return (-1, -1)


def sort_entries_later_first(markdown: str, entries: list[JsonEntry]) -> list[JsonEntry]:
    indexed_entries = list(enumerate(entries))
    indexed_entries.sort(
        key=lambda indexed_entry: (
            markdown_position_for_entry(markdown, indexed_entry[1]),
            -indexed_entry[0],
        ),
        reverse=True,
    )
    return [entry for _index, entry in indexed_entries]


def set_json_value_at_path(root: Any, path: str, value: Any) -> Any:
    if path == "$":
        return value

    parts: list[str | int] = []
    index = 1
    while index < len(path):
        if path[index] == ".":
            index += 1
            start = index
            while index < len(path) and path[index] not in ".[":
                index += 1
            parts.append(path[start:index])
        elif path[index] == "[":
            index += 1
            start = index
            while index < len(path) and path[index] != "]":
                index += 1
            parts.append(int(path[start:index]))
            index += 1
        else:
            raise ValueError(f"Unsupported JSON entry path: {path}")

    target = root
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    return root


class MarkdownJsonEditor:
    def __init__(
        self,
        root: tk.Tk,
        json_path: Path,
        markdown_path: Path,
        backup_enabled: bool = True,
    ) -> None:
        self.root = root
        self.json_path = json_path
        self.markdown_path = markdown_path
        self.backup_enabled = backup_enabled
        self.backup_created = False
        self.json_backup_created = False
        self.dirty = False
        self.json_dirty = False
        self.json_data: Any = None
        self.entries: list[JsonEntry] = []
        self.filtered_entries: list[JsonEntry] = []
        self.current_entry: JsonEntry | None = None
        self.applied_entry_paths: set[str] = set()

        self.search_var = tk.StringVar()
        self.hide_applied_var = tk.BooleanVar(value=True)
        self.hide_non_applicable_var = tk.BooleanVar(value=True)
        self.entry_count_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.root.title("Markdown JSON Editor")
        self.root.geometry("1250x820")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.build_ui()
        self.load_files()

    def build_ui(self) -> None:
        self.build_menu()

        header = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        header.pack(fill=tk.X)
        self.file_label = ttk.Label(header, text="", anchor=tk.W)
        self.file_label.pack(fill=tk.X)

        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(main_pane, width=340)
        main_pane.add(left, weight=1)
        self.build_entry_panel(left)

        right_pane = ttk.Panedwindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=4)

        editor_frame = ttk.Frame(right_pane)
        right_pane.add(editor_frame, weight=4)
        self.build_editor_panel(editor_frame)

        detail_frame = ttk.Frame(right_pane)
        right_pane.add(detail_frame, weight=2)
        self.build_detail_panel(detail_frame)

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=(8, 3),
        )
        status.pack(fill=tk.X)

    def build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Save Markdown", accelerator="Ctrl+S", command=self.save)
        file_menu.add_command(label="Save JSON", command=self.save_json)
        file_menu.add_command(label="Save Markdown As...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Reload Files", accelerator="Ctrl+R", command=self.reload_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)
        menu.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu)

        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Control-r>", lambda _event: self.reload_files())
        self.root.bind("<F3>", lambda _event: self.find_in_markdown())

    def build_entry_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="JSON Entries").pack(anchor=tk.W, pady=(0, 4))

        search = ttk.Entry(parent, textvariable=self.search_var)
        search.pack(fill=tk.X, pady=(0, 6))
        search.insert(0, "")
        self.search_var.trace_add("write", lambda *_args: self.refresh_entry_list())

        filters = ttk.Frame(parent)
        filters.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(
            filters,
            text="Hide applied",
            variable=self.hide_applied_var,
            command=self.refresh_pending,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            filters,
            text="Hide no-longer-applicable",
            variable=self.hide_non_applicable_var,
            command=self.refresh_pending,
        ).pack(anchor=tk.W)
        ttk.Button(filters, text="Refresh Pending", command=self.refresh_pending).pack(
            anchor=tk.W, pady=(4, 0)
        )
        ttk.Label(parent, textvariable=self.entry_count_var, foreground="#555555").pack(
            anchor=tk.W, pady=(0, 6)
        )

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.entry_list = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="dotbox",
            yscrollcommand=scrollbar.set,
        )
        self.entry_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.entry_list.yview)
        self.entry_list.bind("<<ListboxSelect>>", self.on_entry_selected)

        help_text = (
            "Select an entry, review its JSON, then apply the suggested text "
            "with the buttons below."
        )
        ttk.Label(parent, text=help_text, wraplength=320, foreground="#555555").pack(
            fill=tk.X, pady=(8, 0)
        )

    def build_editor_panel(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(toolbar, text="Markdown").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Find", command=self.find_in_markdown).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(toolbar, text="Go To Line", command=self.go_to_line).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(toolbar, text="Save", command=self.save).pack(side=tk.RIGHT, padx=(4, 0))

        editor_wrap = ttk.Frame(parent)
        editor_wrap.pack(fill=tk.BOTH, expand=True)
        y_scroll = ttk.Scrollbar(editor_wrap, orient=tk.VERTICAL)
        x_scroll = ttk.Scrollbar(editor_wrap, orient=tk.HORIZONTAL)
        self.editor = tk.Text(
            editor_wrap,
            wrap=tk.WORD,
            undo=True,
            font=("Consolas", 11),
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        y_scroll.config(command=self.editor.yview)
        x_scroll.config(command=self.editor.xview)
        self.editor.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        editor_wrap.columnconfigure(0, weight=1)
        editor_wrap.rowconfigure(0, weight=1)
        self.editor.tag_configure("json_find", background="#fff2a8")
        self.editor.bind("<<Modified>>", self.on_editor_modified)

    def build_detail_panel(self, parent: ttk.Frame) -> None:
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(buttons, text="Apply Find/Replace", command=self.apply_find_replace).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Apply JSON Edit", command=self.apply_json_edit).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Replace Selection", command=self.replace_selection).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Insert At Cursor", command=self.insert_at_cursor).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Copy Suggestion", command=self.copy_suggestion).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Dismiss Entry", command=self.dismiss_current_entry).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Save JSON", command=self.save_json).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        item_tab = ttk.Frame(notebook)
        find_tab = ttk.Frame(notebook)
        suggestion_tab = ttk.Frame(notebook)
        diff_tab = ttk.Frame(notebook)
        notebook.add(item_tab, text="JSON")
        notebook.add(find_tab, text="Find Text")
        notebook.add(suggestion_tab, text="Suggested Text")
        notebook.add(diff_tab, text="Diff")

        self.item_json = self.create_scrolled_text(item_tab)
        self.find_text = self.create_scrolled_text(find_tab)
        self.suggestion_text = self.create_scrolled_text(suggestion_tab)
        self.find_text.bind("<<Modified>>", self.on_diff_source_modified)
        self.suggestion_text.bind("<<Modified>>", self.on_diff_source_modified)
        self.diff_text = self.create_scrolled_text(diff_tab, readonly=True)
        self.diff_text.tag_configure("diff_delete", background="#ffd7d7")
        self.diff_text.tag_configure("diff_insert", background="#d7ffd9")
        self.diff_text.tag_configure("diff_heading", foreground="#555555")

    def create_scrolled_text(self, parent: ttk.Frame, readonly: bool = False) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            height=7,
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
        )
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        if readonly:
            text.config(state=tk.DISABLED)
        return text

    def load_files(self) -> None:
        try:
            raw_json = self.json_path.read_text(encoding="utf-8")
            json_data = json.loads(raw_json)
            markdown = self.markdown_path.read_text(encoding="utf-8")
        except OSError as error:
            messagebox.showerror("File Error", str(error))
            raise SystemExit(1) from error
        except json.JSONDecodeError as error:
            messagebox.showerror("JSON Error", f"Could not parse JSON:\n{error}")
            raise SystemExit(1) from error

        self.json_data = json_data
        self.entries = sort_entries_later_first(markdown, build_entries(json_data))
        self.applied_entry_paths.clear()
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", markdown)
        self.editor.edit_modified(False)
        self.dirty = False
        self.json_dirty = False
        self.refresh_entry_list()
        self.file_label.config(
            text=f"JSON: {self.json_path}    Markdown: {self.markdown_path}"
        )
        self.status_var.set(
            f"Loaded {len(self.entries)} JSON entr{'y' if len(self.entries) == 1 else 'ies'}"
        )

    def reload_files(self) -> None:
        if self.dirty and not messagebox.askyesno(
            "Discard unsaved edits?",
            "Reloading will discard unsaved Markdown edits. Continue?",
        ):
            return
        self.load_files()

    def current_markdown(self) -> str:
        return self.text_value(self.editor)

    def refresh_pending(self) -> None:
        selected_path = self.current_entry.path if self.current_entry else None
        self.refresh_entry_list(selected_path=selected_path)

    def entry_state(self, entry: JsonEntry, markdown: str | None = None) -> str:
        if entry.path in self.applied_entry_paths:
            return "applied"
        if not entry.find_text:
            return "manual"
        if entry.find_text in (markdown if markdown is not None else self.current_markdown()):
            return "pending"
        return "not_applicable"

    def entry_matches_query(self, entry: JsonEntry, query: str) -> bool:
        if not query:
            return True
        return (
            query in entry.title.lower()
            or query in entry.path.lower()
            or query in json.dumps(entry.data, ensure_ascii=False).lower()
        )

    def refresh_entry_list(self, selected_path: str | None = None) -> None:
        query = self.search_var.get().strip().lower()
        markdown = self.current_markdown()
        states = {entry.path: self.entry_state(entry, markdown) for entry in self.entries}
        self.filtered_entries = []
        for entry in self.entries:
            state = states[entry.path]
            if self.hide_applied_var.get() and state == "applied":
                continue
            if self.hide_non_applicable_var.get() and state == "not_applicable":
                continue
            if not self.entry_matches_query(entry, query):
                continue
            self.filtered_entries.append(entry)

        applied_count = sum(1 for state in states.values() if state == "applied")
        not_applicable_count = sum(
            1 for state in states.values() if state == "not_applicable"
        )
        pending_count = len(self.entries) - applied_count - not_applicable_count
        self.entry_count_var.set(
            f"Showing {len(self.filtered_entries)}/{len(self.entries)} "
            f"({pending_count} pending, {applied_count} applied, "
            f"{not_applicable_count} no longer applicable)"
        )

        self.entry_list.delete(0, tk.END)
        for index, entry in enumerate(self.filtered_entries, start=1):
            prefix = f"{index}. "
            line_hint = f" [line {entry.line_number}]" if entry.line_number else ""
            state = states[entry.path]
            state_hint = ""
            if state == "applied":
                state_hint = " [applied]"
            elif state == "not_applicable":
                state_hint = " [no longer applies]"
            self.entry_list.insert(
                tk.END,
                f"{prefix}{entry.title}{line_hint}{state_hint}",
            )

        if self.filtered_entries:
            selected_index = 0
            if selected_path:
                for index, entry in enumerate(self.filtered_entries):
                    if entry.path == selected_path:
                        selected_index = index
                        break
            self.entry_list.selection_set(selected_index)
            self.entry_list.activate(selected_index)
            self.show_entry(self.filtered_entries[selected_index])
        else:
            self.current_entry = None
            self.set_text(self.item_json, "")
            self.set_text(self.find_text, "")
            self.set_text(self.suggestion_text, "")
            self.render_diff("", "")
            self.status_var.set("No pending JSON entries match the current filters")

    def on_entry_selected(self, _event: tk.Event) -> None:
        selection = self.entry_list.curselection()
        if not selection:
            return
        self.show_entry(self.filtered_entries[selection[0]])

    def show_entry(self, entry: JsonEntry) -> None:
        self.current_entry = entry
        formatted = json.dumps(entry.data, ensure_ascii=False, indent=2)
        self.set_text(self.item_json, formatted)
        self.set_text(self.find_text, entry.find_text)
        self.set_text(self.suggestion_text, entry.replacement_text)
        self.render_diff(entry.find_text, entry.replacement_text)
        if entry.find_text and self.highlight_find_text(
            entry.find_text,
            start_index=f"{entry.line_number}.0" if entry.line_number else "1.0",
            quiet=True,
        ):
            return
        self.status_var.set(f"Selected {entry.path}: {entry.title}")

    def apply_json_edit(self) -> None:
        if not self.current_entry:
            messagebox.showinfo("No Entry", "Select a JSON entry before applying edits.")
            return

        try:
            data = json.loads(self.text_value(self.item_json))
        except json.JSONDecodeError as error:
            messagebox.showerror("JSON Error", f"Could not parse edited JSON:\n{error}")
            return

        updated_entry = make_entry(self.current_entry.path, data)
        try:
            self.json_data = set_json_value_at_path(
                self.json_data,
                self.current_entry.path,
                data,
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            messagebox.showerror("JSON Path Error", str(error))
            return

        for index, entry in enumerate(self.entries):
            if entry.path == self.current_entry.path:
                self.entries[index] = updated_entry
                break

        self.current_entry = updated_entry
        self.json_dirty = True
        self.render_diff(updated_entry.find_text, updated_entry.replacement_text)
        self.refresh_entry_list(selected_path=updated_entry.path)
        self.status_var.set(f"Applied JSON edit to {updated_entry.path}")

    def dismiss_entry(self, entry: JsonEntry, status: str) -> None:
        self.applied_entry_paths.add(entry.path)
        self.refresh_entry_list()
        self.status_var.set(status)

    def dismiss_current_entry(self) -> None:
        if not self.current_entry:
            messagebox.showinfo("No Entry", "Select a JSON entry before dismissing it.")
            return
        self.dismiss_entry(
            self.current_entry,
            f"Dismissed {self.current_entry.path} from the pending list",
        )

    def set_text(self, widget: tk.Text, value: str, readonly: bool = False) -> None:
        if readonly:
            widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        if readonly:
            widget.config(state=tk.DISABLED)

    def insert_diff_piece(self, value: str, tag: str | None = None) -> None:
        if not value:
            return
        start = self.diff_text.index(tk.INSERT)
        self.diff_text.insert(tk.INSERT, value)
        if tag:
            self.diff_text.tag_add(tag, start, self.diff_text.index(tk.INSERT))

    def render_diff(self, find_text: str, replacement_text: str) -> None:
        self.diff_text.config(state=tk.NORMAL)
        self.diff_text.delete("1.0", tk.END)

        if not find_text and not replacement_text:
            self.diff_text.insert("1.0", "No find or replacement text for this entry.")
            self.diff_text.config(state=tk.DISABLED)
            return

        self.diff_text.insert(tk.INSERT, "Current\n", "diff_heading")
        self.render_diff_line(find_text, replacement_text, show_old=True)
        self.diff_text.insert(tk.INSERT, "\n\nProposed\n", "diff_heading")
        self.render_diff_line(find_text, replacement_text, show_old=False)
        self.diff_text.config(state=tk.DISABLED)

    def render_diff_line(
        self,
        find_text: str,
        replacement_text: str,
        show_old: bool,
    ) -> None:
        old_tokens = re.findall(r"\s+|\S+", find_text)
        new_tokens = re.findall(r"\s+|\S+", replacement_text)
        matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)

        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal":
                tokens = old_tokens[old_start:old_end] if show_old else new_tokens[new_start:new_end]
                self.insert_diff_piece("".join(tokens))
            elif show_old and tag in {"delete", "replace"}:
                self.insert_diff_piece("".join(old_tokens[old_start:old_end]), "diff_delete")
            elif not show_old and tag in {"insert", "replace"}:
                self.insert_diff_piece("".join(new_tokens[new_start:new_end]), "diff_insert")

    def on_diff_source_modified(self, event: tk.Event) -> None:
        widget = event.widget
        if isinstance(widget, tk.Text) and widget.edit_modified():
            widget.edit_modified(False)
            self.render_diff(self.text_value(self.find_text), self.text_value(self.suggestion_text))

    def text_value(self, widget: tk.Text) -> str:
        return widget.get("1.0", "end-1c")

    def on_editor_modified(self, _event: tk.Event) -> None:
        if self.editor.edit_modified():
            self.dirty = True
            self.editor.edit_modified(False)
            self.status_var.set("Markdown edited")

    def highlight_find_text(
        self,
        find_text: str,
        start_index: str = "1.0",
        quiet: bool = False,
    ) -> bool:
        if not find_text:
            if not quiet:
                messagebox.showinfo("No Find Text", "This JSON entry has no find/search text.")
            return False

        self.editor.tag_remove("json_find", "1.0", tk.END)
        position = self.editor.search(find_text, start_index, stopindex=tk.END)
        if not position:
            position = self.editor.search(find_text, "1.0", stopindex=tk.END)
        if not position:
            self.status_var.set("Find text not found in Markdown")
            if not quiet:
                self.refresh_pending()
                messagebox.showinfo("Not Found", "The find text was not found in the Markdown file.")
            return False

        end = f"{position}+{len(find_text)}c"
        self.editor.tag_add("json_find", position, end)
        self.editor.mark_set(tk.INSERT, position)
        self.editor.see(position)
        self.editor.focus_set()
        self.status_var.set(f"Found text at {position}")
        return True

    def find_in_markdown(self) -> bool:
        find_text = self.text_value(self.find_text)
        start = self.editor.index("insert + 1c")
        return self.highlight_find_text(find_text, start_index=start, quiet=False)

    def go_to_line(self) -> None:
        line_number = self.current_entry.line_number if self.current_entry else None
        if not line_number:
            if self.find_in_markdown():
                return
            messagebox.showinfo(
                "No Line Number",
                "This JSON entry has no line number and its find text could not be used.",
            )
            return
        index = f"{line_number}.0"
        self.editor.mark_set(tk.INSERT, index)
        self.editor.see(index)
        self.editor.focus_set()
        self.status_var.set(f"Moved to line {line_number}")

    def apply_find_replace(self) -> None:
        find_text = self.text_value(self.find_text)
        replacement = self.text_value(self.suggestion_text)
        if not find_text:
            messagebox.showinfo("No Find Text", "Add find text before applying find/replace.")
            return
        if not replacement:
            if not messagebox.askyesno(
                "Empty Replacement",
                "The replacement text is empty. This will delete the matched text. Continue?",
            ):
                return

        self.editor.tag_remove("json_find", "1.0", tk.END)
        search_start = "1.0"
        if self.current_entry and self.current_entry.line_number:
            search_start = f"{self.current_entry.line_number}.0"
        position = self.editor.search(find_text, search_start, stopindex=tk.END)
        if not position and search_start != "1.0":
            position = self.editor.search(find_text, "1.0", stopindex=tk.END)
        if not position:
            current_entry = self.current_entry
            self.refresh_pending()
            if current_entry and self.entry_state(current_entry) == "not_applicable":
                messagebox.showinfo(
                    "No Longer Applicable",
                    "The find text was not found, so this entry was removed from the pending list.",
                )
                self.status_var.set(
                    f"Removed no-longer-applicable entry {current_entry.path}"
                )
            else:
                messagebox.showinfo("Not Found", "The find text was not found in the Markdown file.")
            return

        current_entry = self.current_entry
        end = f"{position}+{len(find_text)}c"
        self.editor.delete(position, end)
        self.editor.insert(position, replacement)
        self.editor.tag_add("json_find", position, f"{position}+{len(replacement)}c")
        self.editor.mark_set(tk.INSERT, f"{position}+{len(replacement)}c")
        self.editor.see(position)
        self.dirty = True
        if current_entry:
            self.dismiss_entry(
                current_entry,
                f"Applied replacement at {position}; removed entry from pending list",
            )
        else:
            self.refresh_pending()
            self.status_var.set(f"Applied replacement at {position}")

    def replace_selection(self) -> None:
        replacement = self.text_value(self.suggestion_text)
        try:
            start = self.editor.index(tk.SEL_FIRST)
            end = self.editor.index(tk.SEL_LAST)
        except tk.TclError:
            messagebox.showinfo("No Selection", "Select Markdown text before replacing it.")
            return
        current_entry = self.current_entry
        self.editor.delete(start, end)
        self.editor.insert(start, replacement)
        self.editor.mark_set(tk.INSERT, f"{start}+{len(replacement)}c")
        self.editor.see(start)
        self.dirty = True
        if current_entry:
            self.dismiss_entry(
                current_entry,
                f"Replaced selection at {start}; removed entry from pending list",
            )
        else:
            self.refresh_pending()
            self.status_var.set(f"Replaced selection at {start}")

    def insert_at_cursor(self) -> None:
        replacement = self.text_value(self.suggestion_text)
        if not replacement:
            messagebox.showinfo("No Suggested Text", "There is no suggested text to insert.")
            return
        current_entry = self.current_entry
        position = self.editor.index(tk.INSERT)
        self.editor.insert(position, replacement)
        self.editor.mark_set(tk.INSERT, f"{position}+{len(replacement)}c")
        self.editor.see(position)
        self.dirty = True
        if current_entry:
            self.dismiss_entry(
                current_entry,
                f"Inserted suggestion at {position}; removed entry from pending list",
            )
        else:
            self.refresh_pending()
            self.status_var.set(f"Inserted suggestion at {position}")

    def copy_suggestion(self) -> None:
        suggestion = self.text_value(self.suggestion_text)
        self.root.clipboard_clear()
        self.root.clipboard_append(suggestion)
        self.status_var.set("Copied suggestion to clipboard")

    def save(self) -> None:
        if self.backup_enabled and not self.backup_created and self.markdown_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = self.markdown_path.with_name(
                f"{self.markdown_path.name}.{timestamp}.bak"
            )
            backup_path.write_text(self.markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
            self.backup_created = True

        self.markdown_path.write_text(self.text_value(self.editor), encoding="utf-8")
        self.dirty = False
        self.status_var.set(f"Saved Markdown to {self.markdown_path}")

    def save_json(self) -> None:
        if self.json_data is None:
            messagebox.showinfo("No JSON", "No JSON file is loaded.")
            return

        if self.backup_enabled and not self.json_backup_created and self.json_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = self.json_path.with_name(f"{self.json_path.name}.{timestamp}.bak")
            backup_path.write_text(self.json_path.read_text(encoding="utf-8"), encoding="utf-8")
            self.json_backup_created = True

        self.json_path.write_text(
            json.dumps(self.json_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.json_dirty = False
        self.status_var.set(f"Saved JSON to {self.json_path}")

    def save_as(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save Markdown As",
            initialfile=self.markdown_path.name,
            defaultextension=".md",
            filetypes=(('Markdown files', '*.md'), ('Text files', '*.txt'), ('All files', '*.*')),
        )
        if not chosen:
            return
        self.markdown_path = Path(chosen)
        self.backup_created = False
        self.file_label.config(
            text=f"JSON: {self.json_path}    Markdown: {self.markdown_path}"
        )
        self.save()

    def close(self) -> None:
        if self.dirty:
            result = messagebox.askyesnocancel(
                "Unsaved Edits",
                "Save Markdown edits before closing?",
            )
            if result is None:
                return
            if result:
                self.save()
        if self.json_dirty:
            result = messagebox.askyesnocancel(
                "Unsaved JSON Edits",
                "Save JSON edits before closing?",
            )
            if result is None:
                return
            if result:
                self.save_json()
        self.root.destroy()


def choose_file(title: str, filetypes: tuple[tuple[str, str], ...]) -> Path | None:
    chosen = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return Path(chosen) if chosen else None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path] | None:
    json_path = Path(args.json_file) if args.json_file else None
    markdown_path = Path(args.markdown_file) if args.markdown_file else None

    if json_path and markdown_path:
        return json_path, markdown_path

    picker = tk.Tk()
    picker.withdraw()
    try:
        if not json_path:
            json_path = choose_file(
                "Choose JSON file",
                (("JSON files", "*.json"), ("All files", "*.*")),
            )
        if not json_path:
            return None
        if not markdown_path:
            markdown_path = choose_file(
                "Choose Markdown file",
                (("Markdown files", "*.md"), ("Text files", "*.txt"), ("All files", "*.*")),
            )
        if not markdown_path:
            return None
        return json_path, markdown_path
    finally:
        picker.destroy()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a small UI for editing Markdown using JSON suggestions."
    )
    parser.add_argument("json_file", nargs="?", help="Path to a JSON file with edit guidance")
    parser.add_argument("markdown_file", nargs="?", help="Path to the Markdown file to edit")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped .bak file before the first save",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    paths = resolve_paths(args)
    if not paths:
        return 1

    json_path, markdown_path = paths
    root = tk.Tk()
    MarkdownJsonEditor(
        root,
        json_path=json_path,
        markdown_path=markdown_path,
        backup_enabled=not args.no_backup,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
