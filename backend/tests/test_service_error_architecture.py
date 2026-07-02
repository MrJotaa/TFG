import ast
from pathlib import Path


SERVICES_DIRECTORY = Path("app/services")


def _get_service_files() -> list[Path]:
    return sorted(
        path
        for path in SERVICES_DIRECTORY.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _parse_python_file(path: Path) -> ast.AST:
    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )


def test_services_do_not_import_http_exception():
    violations = []

    for service_file in _get_service_files():
        syntax_tree = _parse_python_file(service_file)

        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.module not in {
                "fastapi",
                "fastapi.exceptions",
            }:
                continue

            imported_names = {
                imported_name.name
                for imported_name in node.names
            }

            if "HTTPException" in imported_names:
                violations.append(
                    f"{service_file}:{node.lineno}"
                )

    assert violations == [], (
        "Services must use application errors instead of "
        f"HTTPException: {violations}"
    )


def test_services_do_not_catch_broad_exceptions():
    violations = []

    for service_file in _get_service_files():
        syntax_tree = _parse_python_file(service_file)

        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.ExceptHandler):
                continue

            if node.type is None:
                violations.append(
                    f"{service_file}:{node.lineno} "
                    "contains a bare except"
                )
                continue

            if (
                isinstance(node.type, ast.Name)
                and node.type.id in {
                    "Exception",
                    "BaseException",
                }
            ):
                violations.append(
                    f"{service_file}:{node.lineno} "
                    f"catches {node.type.id}"
                )

    assert violations == [], (
        "Services must not hide unexpected errors with broad "
        f"exception handlers: {violations}"
    )