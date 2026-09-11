"""Guards against training and serving drifting apart again.

The pipeline and the Flask API previously each carried their own copy of
`preprocess_comment`. If they diverge the model is served text shaped
differently from what it was trained on, which degrades predictions with no
visible error. These tests fail if a second copy reappears.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

MODULES_THAT_MUST_SHARE = [
    REPO_ROOT / 'flask_api' / 'main.py',
    REPO_ROOT / 'src' / 'data' / 'data_preprocessing.py',
]


def _parse(path):
    return ast.parse(path.read_text())


def test_preprocessing_is_defined_exactly_once():
    definitions = [
        path for path in REPO_ROOT.glob('**/*.py')
        if '.git' not in path.parts and 'preprocess_comment' in path.read_text()
        and any(isinstance(node, ast.FunctionDef) and node.name == 'preprocess_comment'
                for node in ast.walk(_parse(path)))
    ]

    assert [p.name for p in definitions] == ['preprocessing.py'], (
        f'preprocess_comment should only be defined in src/preprocessing.py, '
        f'found in: {[str(p.relative_to(REPO_ROOT)) for p in definitions]}'
    )


def test_consumers_import_the_shared_implementation():
    for path in MODULES_THAT_MUST_SHARE:
        imports_shared = any(
            isinstance(node, ast.ImportFrom)
            and node.module == 'src.preprocessing'
            and any(alias.name == 'preprocess_comment' for alias in node.names)
            for node in ast.walk(_parse(path))
        )
        assert imports_shared, (
            f'{path.relative_to(REPO_ROOT)} must import preprocess_comment '
            f'from src.preprocessing'
        )


def test_no_hardcoded_infrastructure_addresses():
    """Tracking servers and API hosts belong in config, not in source."""
    offenders = []
    for path in (REPO_ROOT / 'src').glob('**/*.py'):
        text = path.read_text()
        if 'amazonaws.com' in text or 'http://5' in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f'hardcoded infrastructure addresses in: {offenders}'


def test_dvc_stages_run_modules_not_scripts():
    """`python path/to/script.py` puts the script's own directory on sys.path,
    not the repository root, so `from src...` imports fail unless the package
    happens to be pip-installed. `python -m` keeps the root importable."""
    import yaml

    with open(REPO_ROOT / 'dvc.yaml') as f:
        pipeline = yaml.safe_load(f)

    for name, stage in pipeline['stages'].items():
        assert stage['cmd'].startswith('python -m '), (
            f"stage '{name}' invokes a script path; use `python -m` so the "
            f"repository root stays importable"
        )
