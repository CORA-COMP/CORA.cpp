"""test_oracle_pin - the oracle-pin check of scripts/ci/check_oracle_pin.py

Syntax:   python tests/global/test_oracle_pin.py
Checks:   the pinned tag is read from coralean.cmake; the tags come from a `git ls-remote` listing
          (annotated tags peeled, other refs ignored); a later day or a later commit of the same day
          is newer, an earlier one is not.
Exit code: 0 if all pass.

Authors:       Claude Sonnet 5.5
Written:       10-November-2026
Last update:   ---
Reviewed:      ---
"""
import sys
import unittest
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import check_oracle_pin as pin  # noqa: E402


LISTING = "\n".join([
    "aaa\trefs/tags/oracle-20261107-9c0c5b775",
    "bbb\trefs/tags/oracle-20261109-b09283d9c",
    "ccc\trefs/tags/oracle-20261109-f23dcd3c7",
    "ddd\trefs/tags/v1.0",
    "eee\trefs/tags/oracle-20261110-0123456ab",
    "fff\trefs/tags/oracle-20261110-0123456ab^{}"])
DATES = {"aaa": "2026-11-07T10:00:00Z", "bbb": "2026-11-09T16:16:44Z", "ccc": "2026-11-09T20:35:42Z",
         "fff": "2026-11-10T08:00:00Z"}


# ============================================  MAIN  ============================================ #

class OraclePin(unittest.TestCase):
    def test_the_pin_is_read_from_the_cmake_file(self):
        self.assertEqual(pin.pinned('# x\nset(CORALEAN_VERSION "oracle-20261109-b09283d9c")\n'),
                         "oracle-20261109-b09283d9c")
        with self.assertRaises(ValueError):
            pin.pinned("set(OTHER 1)")
        # the committed file names a tag of the published form
        self.assertRegex(pin.pinned((ROOT / "scripts" / "coralean.cmake").read_text()), pin.TAG)

    def test_tags_are_the_oracle_tags_with_peeled_commits(self):
        tags = pin.tags_of(LISTING)
        self.assertEqual(sorted(tags), ["oracle-20261107-9c0c5b775", "oracle-20261109-b09283d9c",
                                        "oracle-20261109-f23dcd3c7", "oracle-20261110-0123456ab"])
        self.assertEqual(tags["oracle-20261110-0123456ab"], "fff")

    def test_a_later_day_or_commit_is_newer(self):
        tags = pin.tags_of(LISTING)
        self.assertEqual(pin.newer_than("oracle-20261109-b09283d9c", tags, DATES.get),
                         ["oracle-20261110-0123456ab", "oracle-20261109-f23dcd3c7"])
        self.assertEqual(pin.newer_than("oracle-20261110-0123456ab", tags, DATES.get), [])
        # the same day, an earlier commit: not newer
        self.assertEqual(pin.newer_than("oracle-20261109-f23dcd3c7", tags, DATES.get),
                         ["oracle-20261110-0123456ab"])


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
