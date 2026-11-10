"""check_oracle_pin - fails when the CORALean oracle pinned in scripts/coralean.cmake is not the newest

Syntax:   python scripts/ci/check_oracle_pin.py
Checks:   the tag CORALEAN_VERSION of scripts/coralean.cmake against the tags `oracle-*` that CORALean
          publishes on its public mirror (git ls-remote); the newest is the one whose commit is the
          latest (the date in the tag name orders days, the commit date orders tags of one day).
Reads:    CORACPP_CORALEAN_REPO (default https://github.com/TUMcps/CORALean), GITHUB_TOKEN (optional,
          for the rate limit of the API that gives the commit dates)
Exit code: 0 if the pin is the newest, 1 with the newer tag (the nightly then opens its issue), 2 if the
          pin or the tags cannot be read.

Authors:       Claude Sonnet 5.5
Written:       10-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

REPO = Path(__file__).resolve().parents[2]
MIRROR = os.environ.get("CORACPP_CORALEAN_REPO", "https://github.com/TUMcps/CORALean")
TAG = re.compile(r"^oracle-(\d{8})-[0-9a-f]+$")


def pinned(text):
    """The tag of `set(CORALEAN_VERSION "...")` in the text of coralean.cmake."""
    found = re.search(r'^set\(CORALEAN_VERSION\s+"([^"]+)"\)', text, re.M)
    if not found:
        raise ValueError("no `set(CORALEAN_VERSION \"...\")` in scripts/coralean.cmake")
    return found.group(1)


def tags_of(listing):
    """`{tag: commit sha}` of the `oracle-*` tags in the output of `git ls-remote --tags`; a peeled
    entry (`^{}`, an annotated tag) gives the commit."""
    out = {}
    for line in listing.splitlines():
        sha, _, ref = line.partition("\t")
        name = ref.removeprefix("refs/tags/").removesuffix("^{}")
        if TAG.match(name) and (ref.endswith("^{}") or name not in out):
            out[name] = sha
    return out


def newer_than(pin, tags, commit_date):
    """The tags later than `pin`, newest first: a later day by the name, the same day by the commit
    date (`commit_date(sha)`, an ISO time)."""
    day = TAG.match(pin).group(1)
    later = [t for t in tags if t != pin and TAG.match(t).group(1) >= day]
    stamp = {t: commit_date(tags[t]) for t in later + [pin]}
    return sorted((t for t in later if stamp[t] > stamp[pin]), key=lambda t: stamp[t], reverse=True)


def commit_date(sha):
    """The commit time of a sha on the mirror, from the GitHub API."""
    api = MIRROR.replace("https://github.com/", "https://api.github.com/repos/") + "/commits/" + sha
    request = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["commit"]["committer"]["date"]


# ============================================  MAIN  ============================================ #

def main():
    # the pin, the published tags, and the tags that are later than the pin
    try:
        pin = pinned((REPO / "scripts" / "coralean.cmake").read_text(encoding="utf-8"))
        listing = subprocess.run(["git", "ls-remote", "--tags", MIRROR, "oracle-*"], text=True,
                                 capture_output=True, check=True, timeout=120).stdout
        tags = tags_of(listing)
        if pin not in tags:
            print(f"the pinned oracle {pin} is not a published tag of {MIRROR}")
            return 2
        # a later tag means the pin is behind
        newer = newer_than(pin, tags, commit_date)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        print(f"cannot check the oracle pin: {error}")
        return 2
    if newer:
        print(f"the pinned oracle {pin} is behind {newer[0]}: bump CORALEAN_VERSION and "
              "CORALEAN_ARCHIVE_SHA256 in scripts/coralean.cmake and rerun the oracle tests")
        return 1
    print(f"the pinned oracle {pin} is the newest")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
