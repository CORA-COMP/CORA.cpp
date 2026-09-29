"""banner - the CORA START and CORA END blocks that every Python example prints

`import cora` calls `install()`; when the script being run is an example (a file named
`example_*.py`), the START block is printed at once and the END block when the interpreter exits.
The examples contain no code for it. A script that ends on an exception prints no END block, as
the C++ examples (src/global/banner.h) do not either.

    ==== CORA START ====   CORA.cpp | <example> | default backend: <name>
    ==== CORA END ====     CORA.cpp | <example> | runtime: <seconds>
"""
import atexit
import os
import sys
import time

WIDTH = 70


def block(title):
    """A line of `=` of the block's width with the title in its middle."""
    text = f" {title} "
    left = (WIDTH - len(text)) // 2
    return "=" * left + text + "=" * (WIDTH - left - len(text))


def install(backend_name):
    """Prints the START block now and registers the END block, if an example is running."""
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    program = os.path.basename(main) if main else ""
    if not program.startswith("example_"):
        return
    start = time.perf_counter()
    failed = []
    print(block("CORA START"))
    print(f"CORA.cpp | {program} | default backend: {backend_name()}")
    print("=" * WIDTH + "\n", flush=True)

    hook = sys.excepthook

    def note_failure(*exception):
        failed.append(True)
        hook(*exception)

    sys.excepthook = note_failure

    def footer():
        if not failed:
            print("\n" + block("CORA END"))
            print(f"CORA.cpp | {program} | runtime: {time.perf_counter() - start:.6g} s")
            print("=" * WIDTH, flush=True)

    atexit.register(footer)
