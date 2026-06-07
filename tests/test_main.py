import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from Main import FileSorter, HistoryStore, SortConfig, main, undo_run


class FileSorterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = SortConfig.defaults()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def sorter(self, **options):
        return FileSorter(self.root, self.root, self.config, **options)

    def test_classifies_extension_and_filename_keyword(self):
        image = self.root / "photo.JPG"
        image.write_bytes(b"image")
        report = self.root / "project-report.txt"
        report.write_text("plain text", encoding="utf-8")

        sorter = self.sorter()

        self.assertEqual(sorter.classify(image)[0], "Images")
        self.assertEqual(sorter.classify(report)[0], "Work")

    def test_preview_does_not_change_filesystem(self):
        source = self.root / "notes.txt"
        source.write_text("hello", encoding="utf-8")

        actions = self.sorter().plan()

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].status, "planned")
        self.assertTrue(source.exists())
        self.assertFalse((self.root / "Documents").exists())

    def test_execute_renames_collision_and_undo_restores_source(self):
        source = self.root / "photo.jpg"
        source.write_bytes(b"new image")
        images = self.root / "Images"
        images.mkdir()
        (images / "photo.jpg").write_bytes(b"existing image")

        sorter = self.sorter()
        actions = sorter.plan()
        record = sorter.execute(actions)

        moved = images / "photo (1).jpg"
        self.assertTrue(moved.exists())
        self.assertFalse(source.exists())
        self.assertEqual(actions[0].status, "moved")

        undo = undo_run(self.root, record["run_id"])

        self.assertEqual(undo["operations"][0]["status"], "restored")
        self.assertTrue(source.exists())
        self.assertFalse(moved.exists())

    def test_identical_destination_is_detected_as_duplicate(self):
        source = self.root / "same.pdf"
        source.write_bytes(b"identical")
        documents = self.root / "Documents"
        documents.mkdir()
        (documents / source.name).write_bytes(b"identical")

        actions = self.sorter().plan()

        self.assertEqual(actions[0].status, "duplicate")
        self.assertTrue(source.exists())

    def test_recursive_scan_skips_managed_output_directories(self):
        nested = self.root / "inbox" / "nested.txt"
        nested.parent.mkdir()
        nested.write_text("nested", encoding="utf-8")
        managed = self.root / "Documents" / "already-sorted.txt"
        managed.parent.mkdir()
        managed.write_text("sorted", encoding="utf-8")

        actions = self.sorter(recursive=True).plan()

        self.assertEqual([action.source for action in actions], [nested.resolve()])

    def test_content_keyword_rule(self):
        source = self.root / "notes.txt"
        source.write_text("Agenda for the quarterly discussion", encoding="utf-8")

        actions = self.sorter(inspect_content=True).plan()

        self.assertEqual(actions[0].category, "Work")
        self.assertIn("content keyword", actions[0].reason)

    def test_month_grouping_uses_modified_time(self):
        source = self.root / "photo.png"
        source.write_bytes(b"image")
        timestamp = 1_704_067_200  # 2024-01-01 00:00:00 UTC
        os.utime(source, (timestamp, timestamp))

        action = self.sorter(group_by="month").plan()[0]

        modified = __import__("datetime").datetime.fromtimestamp(timestamp)
        self.assertEqual(
            action.destination,
            self.root / "Images" / f"{modified.year:04d}" / f"{modified.month:02d}" / "photo.png",
        )

    def test_custom_config_merges_rules(self):
        path = self.root / "config.json"
        path.write_text(
            json.dumps(
                {
                    "categories": {"Books": ["epub"], "Documents": ["py"]},
                    "keyword_rules": {"Finance": ["budget"]},
                    "ignore_patterns": ["*.ignore"],
                }
            ),
            encoding="utf-8",
        )

        config = SortConfig.load(path)

        self.assertEqual(config.categories["Books"], (".epub",))
        self.assertEqual(config.categories["Documents"], (".py",))
        self.assertEqual(config.keyword_rules["Finance"], ("budget",))
        self.assertEqual(config.ignore_patterns, ("*.ignore",))

        book = self.root / "novel.epub"
        book.write_bytes(b"book")
        sorter = FileSorter(self.root, self.root, config)
        self.assertEqual(sorter.classify(book)[0], "Books")
        script = self.root / "custom.py"
        script.write_text("pass", encoding="utf-8")
        self.assertEqual(sorter.classify(script)[0], "Documents")

    def test_keyword_matching_does_not_match_inside_larger_word(self):
        source = self.root / "syntax-notes.py"
        source.write_text("parser", encoding="utf-8")

        self.assertEqual(self.sorter().classify(source)[0], "Code")

    def test_json_undo_returns_success(self):
        source = self.root / "photo.png"
        source.write_bytes(b"image")
        sorter = self.sorter()
        sorter.execute(sorter.plan())

        output = StringIO()
        with redirect_stdout(output):
            exit_code = main([str(self.root), "--undo", "--json"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["type"], "undo")
        self.assertTrue(source.exists())

    def test_history_latest_ignores_already_undone_run(self):
        source = self.root / "data.csv"
        source.write_text("a,b", encoding="utf-8")
        sorter = self.sorter()
        record = sorter.execute(sorter.plan())
        undo_run(self.root, "latest")

        latest = HistoryStore(self.root).find_sort_run("latest")

        self.assertIsNone(latest)
        self.assertEqual(
            HistoryStore(self.root).find_sort_run(record["run_id"])["run_id"],
            record["run_id"],
        )


if __name__ == "__main__":
    unittest.main()
