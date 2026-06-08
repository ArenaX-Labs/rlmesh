from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

SKIPPED_PATH_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}


def test_type_checking_only_imports_are_not_used_at_runtime() -> None:
    findings = [
        f"{path}:{line}: {name}" for path, line, name in _type_checking_runtime_uses()
    ]

    assert findings == []


def _type_checking_runtime_uses() -> list[tuple[Path, int, str]]:
    repo_root = _repo_root()
    findings: list[tuple[Path, int, str]] = []

    for path in _python_files(repo_root):
        tree = ast.parse(path.read_text(), filename=str(path))
        parent_by_node = _parent_map(tree)

        type_only_names = _type_checking_imports(tree)
        if not type_only_names:
            continue

        runtime_module_names = _BoundNameCollector.collect(tree)
        risky_names = type_only_names - runtime_module_names
        if not risky_names:
            continue

        annotations_are_deferred = _has_future_annotations(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if node.id not in risky_names:
                continue
            if _inside_type_checking_block(node, parent_by_node):
                continue
            if annotations_are_deferred and _inside_annotation(node, parent_by_node):
                continue
            scope = _enclosing_runtime_scope(node, parent_by_node)
            if scope is not None and node.id in _BoundNameCollector.collect(scope):
                continue
            findings.append((path.relative_to(repo_root), node.lineno, node.id))

    return findings


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "Cargo.toml").exists():
            return parent
    raise AssertionError("could not locate repository root")


def _python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in ("python", "tools", "examples"):
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if SKIPPED_PATH_PARTS.isdisjoint(path.parts):
                files.append(path)
    return sorted(files)


def _has_future_annotations(tree: ast.Module) -> bool:
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            if any(alias.name == "annotations" for alias in statement.names):
                return True
    return False


def _is_type_checking_test(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


def _is_type_checking_if(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and _is_type_checking_test(node.test)


def _imported_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names if alias.name != "*"}
    return set()


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, ast.Tuple | ast.List):
        return {name for child in node.elts for name in _target_names(child)}
    return set()


def _type_checking_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for statement in tree.body:
        if not _is_type_checking_if(statement):
            continue
        for child in ast.walk(statement):
            names.update(_imported_names(child))
    return names


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _inside_type_checking_block(
    node: ast.AST, parent_by_node: Mapping[ast.AST, ast.AST]
) -> bool:
    current = node
    while current in parent_by_node:
        parent = parent_by_node[current]
        if _is_type_checking_if(parent):
            return True
        current = parent
    return False


def _inside_annotation(
    node: ast.AST, parent_by_node: Mapping[ast.AST, ast.AST]
) -> bool:
    current = node
    while current in parent_by_node:
        parent = parent_by_node[current]
        if isinstance(parent, ast.arg) and parent.annotation is current:
            return True
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
            if parent.returns is current:
                return True
        if isinstance(parent, ast.AnnAssign) and parent.annotation is current:
            return True
        current = parent
    return False


def _enclosing_runtime_scope(
    node: ast.AST, parent_by_node: Mapping[ast.AST, ast.AST]
) -> ast.AST | None:
    current = node
    while current in parent_by_node:
        parent = parent_by_node[current]
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return parent
        current = parent
    return None


class _BoundNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    @classmethod
    def collect(cls, node: ast.AST) -> set[str]:
        collector = cls()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            collector.visit(node.args)
            for statement in node.body:
                collector.visit(statement)
        elif isinstance(node, ast.ClassDef):
            for statement in node.body:
                collector.visit(statement)
        else:
            collector.visit(node)
        return collector.names

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking_if(node):
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(_imported_names(node))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(_imported_names(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _ = node

    def visit_arguments(self, node: ast.arguments) -> None:
        for arg in [*node.posonlyargs, *node.args, *node.kwonlyargs]:
            self.names.add(arg.arg)
        if node.vararg is not None:
            self.names.add(node.vararg.arg)
        if node.kwarg is not None:
            self.names.add(node.kwarg.arg)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.names.update(_target_names(target))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.names.update(_target_names(node.target))
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.names.update(_target_names(node.target))
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.names.update(_target_names(node.target))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(_target_names(item.optional_vars))
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(_target_names(item.optional_vars))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)
