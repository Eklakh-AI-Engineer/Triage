"""Keep Kaggle long-running stages fail-fast and output-checked."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "deploy" / "kaggle_triage.ipynb"


def test_notebook_checks_long_running_commands_and_full_test_suite():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {i: "".join(notebook["cells"][i]["source"])
               for i in (1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)}

    for cell, source in sources.items():
        assert "run_checked(" in source, f"cell {cell} bypasses checked subprocesses"
        assert not any(line.lstrip().startswith("!") for line in source.splitlines()), \
            f"cell {cell} contains an unchecked notebook shell command"
    assert "'pytest', 'tests', '-q'" in sources[8]
    assert "'pull', '--rebase'" in sources[1]
    assert "fresh results" in sources[14]
    assert "run_manifest" in sources[14]
    assert "fresh checkpoint" in sources[13]
