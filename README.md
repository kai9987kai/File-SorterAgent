# File Sorter Agent

File Sorter Agent is a safe, deterministic command-line organizer for Windows,
macOS, and Linux. It previews every change, handles naming conflicts, detects
duplicates, records applied runs, and can undo them.

The original TensorFlow reinforcement-learning prototype is retained in
`beta.py` for experimentation. It is not used by the production sorter because
an untrained model makes random file-placement decisions.

## Highlights

- Safe preview mode by default; files move only with `--execute`
- 10 built-in categories covering common documents, media, code, data, apps,
  design files, and fonts
- Filename and optional text/DOCX content rules for `Work` and `Personal`
- Recursive operation with managed output folders automatically excluded
- Collision-safe renaming and SHA-256 duplicate detection
- Optional year or year/month grouping
- JSON configuration, JSON output, ignore patterns, and minimum file age
- Persistent run history and reversible moves
- No required third-party Python packages

## Requirements

Python 3.10 or newer.

## Quick Start

Preview a Downloads folder:

```powershell
python Main.py "$HOME\Downloads"
```

Apply exactly that style of plan:

```powershell
python Main.py "$HOME\Downloads" --execute
```

The preview is recalculated when `--execute` runs, so review the output again if
the directory changed between commands.

## Common Workflows

Sort nested files and group them by year and month:

```powershell
python Main.py "$HOME\Downloads" --recursive --group-by month --execute
```

Inspect text and DOCX content for work/personal keywords:

```powershell
python Main.py "$HOME\Downloads" --content
```

Place organized files under a separate destination:

```powershell
python Main.py "D:\Inbox" --destination "D:\Organized" --execute
```

Skip recently modified files, useful when downloads may still be active:

```powershell
python Main.py "$HOME\Downloads" --min-age 300 --execute
```

Emit machine-readable output:

```powershell
python Main.py "$HOME\Downloads" --json
```

## Undo And History

Every executed run is written to:

```text
<destination>/.file-sorter/history.jsonl
```

Undo the latest run:

```powershell
python Main.py "$HOME\Downloads" --undo
```

Undo a specific run:

```powershell
python Main.py "$HOME\Downloads" --undo 20260607T120000Z-ab12cd34
```

Show available run IDs:

```powershell
python Main.py "$HOME\Downloads" --history
```

Undo never overwrites a file. If the original path is occupied or a sorted file
was moved again, that item is reported as an error and left untouched.

## Duplicate And Collision Handling

Identical destination files are skipped by default:

```powershell
python Main.py . --duplicates skip
```

Alternatives are `--duplicates rename` and `--duplicates folder`. Different
files with the same name receive `name (1).ext` by default. Use
`--collision skip` to leave those source files untouched instead.

## Custom Rules

Generate a complete editable configuration:

```powershell
python Main.py --init-config sorter-config.json
```

Then run with it:

```powershell
python Main.py "$HOME\Downloads" --config sorter-config.json
```

Configuration values merge with the built-in rules. Reusing a category name
replaces that category's values:

```json
{
  "categories": {
    "Design": [".fig", ".sketch", ".xd"],
    "Books": [".epub", ".mobi"]
  },
  "keyword_rules": {
    "Finance": ["invoice", "statement", "budget"]
  },
  "ignore_patterns": ["*.tmp", "keep/**"],
  "content_read_limit": 256000
}
```

Category names cannot contain path separators. Keyword rules take priority over
extension rules.

## Testing

```powershell
python -m unittest discover -s tests -v
```

## Legacy Experiment

`beta.py` contains the previous Dueling DQN and commander-agent experiment. It
requires NumPy, TensorFlow, and python-docx:

```powershell
pip install numpy tensorflow python-docx
python beta.py --directory path\to\sandbox
```

Use it only on disposable test files: its untrained policy may classify files
incorrectly.
