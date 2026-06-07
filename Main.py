from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION = "2.0.0"
HISTORY_DIRECTORY = ".file-sorter"
HISTORY_FILE = "history.jsonl"

DEFAULT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Documents": (
        ".csv", ".doc", ".docx", ".epub", ".key", ".md", ".mobi", ".odt",
        ".ods", ".odp", ".pages", ".pdf", ".ppt", ".pptx", ".rtf", ".tex",
        ".txt", ".xls", ".xlsx",
    ),
    "Images": (
        ".avif", ".bmp", ".cr2", ".dng", ".gif", ".heic", ".ico", ".jpeg",
        ".jpg", ".nef", ".png", ".raw", ".svg", ".tif", ".tiff", ".webp",
    ),
    "Videos": (
        ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
        ".mpg", ".mts", ".vob", ".webm", ".wmv",
    ),
    "Audio": (
        ".aac", ".aiff", ".flac", ".m4a", ".mid", ".midi", ".mp3", ".ogg",
        ".opus", ".wav", ".wma",
    ),
    "Archives": (
        ".7z", ".bz2", ".cab", ".gz", ".iso", ".rar", ".tar", ".tgz", ".xz",
        ".zip",
    ),
    "Code": (
        ".bat", ".c", ".cmd", ".cpp", ".cs", ".css", ".go", ".h", ".html",
        ".java", ".js", ".jsx", ".json", ".lua", ".php", ".ps1", ".py", ".rb",
        ".rs", ".scss", ".sh", ".sql", ".ts", ".tsx", ".vue", ".xml",
    ),
    "Data": (
        ".db", ".feather", ".h5", ".hdf5", ".npy", ".parquet", ".pkl",
        ".rdata", ".sav", ".sqlite", ".yaml", ".yml",
    ),
    "Applications": (".appimage", ".apk", ".dmg", ".exe", ".jar", ".msi"),
    "Design": (
        ".3ds", ".ai", ".blend", ".dae", ".dwg", ".dxf", ".fbx", ".fig",
        ".gltf", ".glb", ".obj", ".psd", ".sketch", ".step", ".stl",
    ),
    "Fonts": (".eot", ".otf", ".ttf", ".woff", ".woff2"),
}

DEFAULT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Work": ("agenda", "client", "contract", "invoice", "meeting", "project", "report"),
    "Personal": ("family", "holiday", "personal", "receipt", "tax", "travel"),
}

DEFAULT_IGNORE_PATTERNS = (
    "*.crdownload",
    "*.download",
    "*.part",
    "~$*",
)

TEXT_EXTENSIONS = {
    ".csv", ".css", ".html", ".ini", ".js", ".json", ".log", ".md", ".py",
    ".rtf", ".sql", ".tex", ".ts", ".txt", ".xml", ".yaml", ".yml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_extension(extension: str) -> str:
    extension = extension.strip().lower()
    return extension if extension.startswith(".") else f".{extension}"


def validate_category(category: str) -> None:
    if (
        not category.strip()
        or category in {".", ".."}
        or "/" in category
        or "\\" in category
        or Path(category).is_absolute()
    ):
        raise ValueError(f"Invalid category name: {category!r}")


@dataclass
class SortConfig:
    categories: dict[str, tuple[str, ...]]
    keyword_rules: dict[str, tuple[str, ...]]
    ignore_patterns: tuple[str, ...]
    content_read_limit: int = 256_000

    @classmethod
    def defaults(cls) -> "SortConfig":
        return cls(
            categories=dict(DEFAULT_CATEGORIES),
            keyword_rules=dict(DEFAULT_KEYWORDS),
            ignore_patterns=DEFAULT_IGNORE_PATTERNS,
        )

    @classmethod
    def load(cls, path: Path | None) -> "SortConfig":
        config = cls.defaults()
        if path is None:
            return config

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a JSON object")

        if "categories" in raw:
            config.categories = _merge_rule_map(config.categories, raw["categories"], "categories")
        if "keyword_rules" in raw:
            config.keyword_rules = _merge_rule_map(
                config.keyword_rules, raw["keyword_rules"], "keyword_rules", extensions=False
            )
        if "ignore_patterns" in raw:
            if not isinstance(raw["ignore_patterns"], list):
                raise ValueError("'ignore_patterns' must be a list")
            config.ignore_patterns = tuple(str(pattern) for pattern in raw["ignore_patterns"])
        if "content_read_limit" in raw:
            config.content_read_limit = int(raw["content_read_limit"])
            if config.content_read_limit < 1:
                raise ValueError("'content_read_limit' must be positive")

        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": {name: list(values) for name, values in self.categories.items()},
            "keyword_rules": {name: list(values) for name, values in self.keyword_rules.items()},
            "ignore_patterns": list(self.ignore_patterns),
            "content_read_limit": self.content_read_limit,
        }


def _merge_rule_map(
    existing: dict[str, tuple[str, ...]],
    new_rules: Any,
    field_name: str,
    *,
    extensions: bool = True,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(new_rules, dict):
        raise ValueError(f"'{field_name}' must be an object")
    merged = dict(existing)
    for category, values in new_rules.items():
        validate_category(category)
        if not isinstance(values, list):
            raise ValueError(f"Rule values for {category!r} must be a list")
        normalized = [
            normalize_extension(str(value)) if extensions else str(value).strip().lower()
            for value in values
            if str(value).strip()
        ]
        # Reinsert customized categories last so their rules override defaults.
        merged.pop(category, None)
        merged[category] = tuple(dict.fromkeys(normalized))
    return merged


@dataclass
class SortAction:
    source: Path
    destination: Path | None
    category: str
    reason: str
    size: int
    status: str = "planned"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "destination": str(self.destination) if self.destination else None,
            "category": self.category,
            "reason": self.reason,
            "size": self.size,
            "status": self.status,
            "error": self.error,
        }


class HistoryStore:
    def __init__(self, destination_root: Path):
        self.directory = destination_root / HISTORY_DIRECTORY
        self.path = self.directory / HISTORY_FILE

    def append(self, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid history entry on line {line_number}") from exc
        return records

    def find_sort_run(self, run_id: str) -> dict[str, Any] | None:
        records = self.records()
        undone = {
            record.get("target_run_id")
            for record in records
            if record.get("type") == "undo"
        }
        for record in reversed(records):
            if record.get("type") != "sort":
                continue
            if run_id == "latest" and record.get("run_id") not in undone:
                return record
            if record.get("run_id") == run_id:
                return record
        return None


class FileSorter:
    def __init__(
        self,
        source_root: Path,
        destination_root: Path,
        config: SortConfig,
        *,
        recursive: bool = False,
        include_hidden: bool = False,
        inspect_content: bool = False,
        group_by: str = "none",
        min_age: float = 0,
        collision: str = "rename",
        duplicate_action: str = "skip",
    ):
        self.source_root = source_root.expanduser().resolve()
        self.destination_root = destination_root.expanduser().resolve()
        self.config = config
        self.recursive = recursive
        self.include_hidden = include_hidden
        self.inspect_content = inspect_content
        self.group_by = group_by
        self.min_age = min_age
        self.collision = collision
        self.duplicate_action = duplicate_action
        self._extension_map = self._build_extension_map()
        self._managed_directories = {
            *config.categories,
            *config.keyword_rules,
            "Other",
            "Duplicates",
            HISTORY_DIRECTORY,
        }

    def _build_extension_map(self) -> dict[str, str]:
        extension_map: dict[str, str] = {}
        for category, extensions in self.config.categories.items():
            validate_category(category)
            for extension in extensions:
                extension_map[normalize_extension(extension)] = category
        for category in self.config.keyword_rules:
            validate_category(category)
        return extension_map

    def classify(self, path: Path) -> tuple[str, str]:
        searchable_name = re.sub(r"[_\-.]+", " ", path.stem.lower())
        for category, keywords in self.config.keyword_rules.items():
            matched = self._find_keyword(searchable_name, keywords)
            if matched:
                return category, f"filename keyword '{matched}'"

        if self.inspect_content:
            content = self._read_searchable_content(path).lower()
            for category, keywords in self.config.keyword_rules.items():
                matched = self._find_keyword(content, keywords)
                if matched:
                    return category, f"content keyword '{matched}'"

        extension = path.suffix.lower()
        category = self._extension_map.get(extension)
        if category:
            return category, f"extension '{extension}'"
        return "Other", "unmatched extension"

    @staticmethod
    def _find_keyword(content: str, keywords: tuple[str, ...]) -> str | None:
        for keyword in keywords:
            if keyword and re.search(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                content,
                flags=re.IGNORECASE,
            ):
                return keyword
        return None

    def _read_searchable_content(self, path: Path) -> str:
        try:
            if path.suffix.lower() in TEXT_EXTENSIONS:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    return handle.read(self.config.content_read_limit)
            if path.suffix.lower() == ".docx":
                with zipfile.ZipFile(path) as archive:
                    with archive.open("word/document.xml") as document:
                        xml = document.read(self.config.content_read_limit)
                text = re.sub(rb"<[^>]+>", b" ", xml).decode("utf-8", errors="ignore")
                return html.unescape(text)
        except (OSError, KeyError, zipfile.BadZipFile):
            return ""
        return ""

    def iter_files(self) -> Iterable[Path]:
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_root}")

        if not self.recursive:
            candidates = (path for path in self.source_root.iterdir() if path.is_file())
        else:
            candidates = self._walk_files()

        now = datetime.now().timestamp()
        script_path = Path(__file__).resolve()
        for path in candidates:
            resolved = path.resolve()
            if resolved == script_path or self._is_ignored(path):
                continue
            try:
                if self.min_age and now - path.stat().st_mtime < self.min_age:
                    continue
            except OSError:
                continue
            yield path

    def _walk_files(self) -> Iterable[Path]:
        for current, directories, files in os.walk(self.source_root):
            current_path = Path(current)
            kept_directories = []
            for directory in directories:
                candidate = current_path / directory
                if self._should_skip_directory(candidate):
                    continue
                kept_directories.append(directory)
            directories[:] = kept_directories
            for filename in files:
                yield current_path / filename

    def _should_skip_directory(self, path: Path) -> bool:
        if path.name == HISTORY_DIRECTORY:
            return True
        if not self.include_hidden and path.name.startswith("."):
            return True
        if self.destination_root == self.source_root and path.parent == self.source_root:
            if path.name in self._managed_directories:
                return True
        try:
            if self.destination_root != self.source_root:
                path.resolve().relative_to(self.destination_root)
                return True
        except ValueError:
            pass
        return self._matches_ignore(path)

    def _is_ignored(self, path: Path) -> bool:
        if not self.include_hidden:
            try:
                relative_parts = path.relative_to(self.source_root).parts
            except ValueError:
                relative_parts = path.parts
            if any(part.startswith(".") for part in relative_parts):
                return True
        return self._matches_ignore(path)

    def _matches_ignore(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.source_root).as_posix()
        except ValueError:
            relative = path.name
        return any(
            fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(relative, pattern)
            for pattern in self.config.ignore_patterns
        )

    def plan(self) -> list[SortAction]:
        actions: list[SortAction] = []
        reserved: set[str] = set()
        for source in sorted(self.iter_files(), key=lambda item: str(item).casefold()):
            try:
                stat = source.stat()
                category, reason = self.classify(source)
                destination = self._destination_for(source, category, stat.st_mtime)
                destination, status, error = self._resolve_destination(
                    source, destination, category, stat.st_mtime, reserved
                )
                if destination is not None:
                    reserved.add(os.path.normcase(str(destination)))
                actions.append(
                    SortAction(
                        source=source.resolve(),
                        destination=destination,
                        category=category,
                        reason=reason,
                        size=stat.st_size,
                        status=status,
                        error=error,
                    )
                )
            except OSError as exc:
                actions.append(
                    SortAction(source, None, "Unknown", "filesystem error", 0, "error", str(exc))
                )
        return actions

    def _destination_for(self, source: Path, category: str, modified_at: float) -> Path:
        destination = self.destination_root / category
        modified = datetime.fromtimestamp(modified_at)
        if self.group_by == "year":
            destination /= f"{modified.year:04d}"
        elif self.group_by == "month":
            destination = destination / f"{modified.year:04d}" / f"{modified.month:02d}"
        return destination / source.name

    def _resolve_destination(
        self,
        source: Path,
        destination: Path,
        category: str,
        modified_at: float,
        reserved: set[str],
    ) -> tuple[Path | None, str, str | None]:
        destination_key = os.path.normcase(str(destination))
        if not destination.exists() and destination_key not in reserved:
            return destination, "planned", None

        if destination.exists() and self._same_content(source, destination):
            if self.duplicate_action == "skip":
                return destination, "duplicate", "identical file already exists"
            if self.duplicate_action == "folder":
                duplicate = self._destination_for(source, "Duplicates", modified_at)
                duplicate = duplicate.parent / category / duplicate.name
                return self._unique_destination(duplicate, reserved), "planned", None

        if self.collision == "skip":
            return destination, "collision", "destination already exists"
        return self._unique_destination(destination, reserved), "planned", None

    @staticmethod
    def _same_content(first: Path, second: Path) -> bool:
        try:
            if first.stat().st_size != second.stat().st_size:
                return False
            return FileSorter._digest(first) == FileSorter._digest(second)
        except OSError:
            return False

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _unique_destination(destination: Path, reserved: set[str]) -> Path:
        counter = 1
        candidate = destination
        while candidate.exists() or os.path.normcase(str(candidate)) in reserved:
            candidate = destination.with_name(
                f"{destination.stem} ({counter}){destination.suffix}"
            )
            counter += 1
        return candidate

    def execute(self, actions: list[SortAction]) -> dict[str, Any]:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        for action in actions:
            if action.status != "planned" or action.destination is None:
                continue
            try:
                if not action.source.exists():
                    raise FileNotFoundError("source no longer exists")
                destination = action.destination
                if destination.exists():
                    destination = self._unique_destination(destination, set())
                    action.destination = destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(action.source), str(destination))
                action.status = "moved"
            except OSError as exc:
                action.status = "error"
                action.error = str(exc)

        record = {
            "version": 1,
            "type": "sort",
            "run_id": run_id,
            "created_at": utc_now(),
            "source_root": str(self.source_root),
            "destination_root": str(self.destination_root),
            "options": {
                "recursive": self.recursive,
                "group_by": self.group_by,
                "inspect_content": self.inspect_content,
            },
            "operations": [action.to_dict() for action in actions],
        }
        HistoryStore(self.destination_root).append(record)
        return record


def undo_run(destination_root: Path, requested_run_id: str) -> dict[str, Any]:
    destination_root = destination_root.expanduser().resolve()
    history = HistoryStore(destination_root)
    target = history.find_sort_run(requested_run_id)
    if target is None:
        raise ValueError(f"No undoable sort run found for {requested_run_id!r}")

    results = []
    for operation in reversed(target.get("operations", [])):
        if operation.get("status") != "moved":
            continue
        source = Path(operation["source"])
        destination = Path(operation["destination"])
        result = {
            "source": str(destination),
            "destination": str(source),
            "status": "restored",
            "error": None,
        }
        try:
            if not destination.exists():
                raise FileNotFoundError("sorted file no longer exists")
            if source.exists():
                raise FileExistsError("original path is occupied")
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
        except OSError as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)

    record = {
        "version": 1,
        "type": "undo",
        "run_id": f"undo-{uuid.uuid4().hex[:8]}",
        "target_run_id": target["run_id"],
        "created_at": utc_now(),
        "operations": results,
    }
    history.append(record)
    return record


def summarize_actions(actions: list[SortAction]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    categories: dict[str, int] = {}
    total_bytes = 0
    for action in actions:
        counts[action.status] = counts.get(action.status, 0) + 1
        categories[action.category] = categories.get(action.category, 0) + 1
        if action.status in {"planned", "moved"}:
            total_bytes += action.size
    return {
        "total": len(actions),
        "counts": counts,
        "categories": categories,
        "bytes": total_bytes,
    }


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_plan(actions: list[SortAction], source_root: Path, *, executed: bool) -> None:
    if not actions:
        print("No eligible files found.")
        return
    for action in actions:
        try:
            source = action.source.relative_to(source_root.resolve())
        except ValueError:
            source = action.source
        destination = action.destination or Path("-")
        marker = {
            "planned": "PLAN",
            "moved": "MOVE",
            "duplicate": "SKIP",
            "collision": "SKIP",
            "error": "ERR ",
        }.get(action.status, action.status.upper())
        print(f"[{marker}] {source} -> {destination} ({action.reason})")
        if action.error:
            print(f"       {action.error}")

    summary = summarize_actions(actions)
    verb = "Processed" if executed else "Previewed"
    print(
        f"\n{verb} {summary['total']} file(s), "
        f"{human_size(summary['bytes'])} eligible to move."
    )
    print("Categories: " + ", ".join(
        f"{category}={count}" for category, count in sorted(summary["categories"].items())
    ))


def print_history(destination_root: Path, *, as_json: bool) -> None:
    records = HistoryStore(destination_root.resolve()).records()
    if as_json:
        print(json.dumps(records, indent=2))
        return
    if not records:
        print("No history found.")
        return
    for record in records:
        operations = record.get("operations", [])
        successful = sum(
            item.get("status") in {"moved", "restored"} for item in operations
        )
        target = f" target={record['target_run_id']}" if record.get("target_run_id") else ""
        print(
            f"{record.get('created_at', '?')}  {record.get('type', '?'):4}  "
            f"{record.get('run_id', '?')}  successful={successful}{target}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely organize files with deterministic, reversible rules."
    )
    parser.add_argument("directory", nargs="?", default=".", help="directory to organize")
    parser.add_argument(
        "--destination",
        type=Path,
        help="destination root (default: organize inside the source directory)",
    )
    parser.add_argument("--execute", action="store_true", help="apply the previewed moves")
    parser.add_argument("--recursive", action="store_true", help="include nested directories")
    parser.add_argument("--include-hidden", action="store_true", help="include hidden files")
    parser.add_argument(
        "--content",
        action="store_true",
        help="inspect text and DOCX content for keyword rules",
    )
    parser.add_argument(
        "--group-by",
        choices=("none", "year", "month"),
        default="none",
        help="create date-based subdirectories",
    )
    parser.add_argument(
        "--min-age",
        type=float,
        default=0,
        metavar="SECONDS",
        help="ignore files modified more recently than this",
    )
    parser.add_argument(
        "--collision",
        choices=("rename", "skip"),
        default="rename",
        help="handling for different files with the same name",
    )
    parser.add_argument(
        "--duplicates",
        choices=("skip", "rename", "folder"),
        default="skip",
        help="handling for identical files at the destination",
    )
    parser.add_argument("--config", type=Path, help="JSON rules file")
    parser.add_argument(
        "--init-config",
        type=Path,
        metavar="PATH",
        help="write an editable default configuration and exit",
    )
    parser.add_argument(
        "--undo",
        nargs="?",
        const="latest",
        metavar="RUN_ID",
        help="undo the latest run, or the supplied run ID",
    )
    parser.add_argument("--history", action="store_true", help="show run history and exit")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_root = Path(args.directory).expanduser().resolve()
    destination_root = (
        args.destination.expanduser().resolve() if args.destination else source_root
    )

    try:
        if args.init_config:
            output = args.init_config.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(SortConfig.defaults().to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote configuration: {output}")
            return 0

        if args.history:
            print_history(destination_root, as_json=args.json)
            return 0

        if args.undo:
            record = undo_run(destination_root, args.undo)
            restored = sum(
                operation["status"] == "restored"
                for operation in record["operations"]
            )
            errors = len(record["operations"]) - restored
            if args.json:
                print(json.dumps(record, indent=2))
            else:
                print(
                    f"Undid {record['target_run_id']}: restored={restored}, errors={errors}"
                )
                for operation in record["operations"]:
                    if operation["error"]:
                        print(f"[ERR ] {operation['source']}: {operation['error']}")
            return 0 if not errors else 1

        config = SortConfig.load(args.config)
        sorter = FileSorter(
            source_root,
            destination_root,
            config,
            recursive=args.recursive,
            include_hidden=args.include_hidden,
            inspect_content=args.content,
            group_by=args.group_by,
            min_age=args.min_age,
            collision=args.collision,
            duplicate_action=args.duplicates,
        )
        actions = sorter.plan()
        record = sorter.execute(actions) if args.execute else None

        if args.json:
            payload = {
                "mode": "execute" if args.execute else "preview",
                "summary": summarize_actions(actions),
                "actions": [action.to_dict() for action in actions],
                "run_id": record["run_id"] if record else None,
            }
            print(json.dumps(payload, indent=2))
        else:
            print_plan(actions, source_root, executed=args.execute)
            if record:
                print(f"Run ID: {record['run_id']}")
                print(f"Undo with: {Path(sys.argv[0]).name} --undo {record['run_id']}")
            elif actions:
                print("\nPreview only. Add --execute to apply these moves.")
        return 1 if any(action.status == "error" for action in actions) else 0
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
