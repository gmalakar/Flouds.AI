# =============================================================================
# File: file_audit.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""File operation audit utilities."""

import ast
import os
from typing import List, Tuple

from app.logger import get_logger

logger = get_logger("file_audit")


class FileOperationVisitor(ast.NodeVisitor):
    """AST visitor to find file operations."""

    def __init__(self):
        self.file_operations = []
        self.current_file = None

    def visit_Call(self, node):
        """Visit function calls to detect file operations."""
        # Check for open() calls
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self.file_operations.append(
                {
                    "type": "open",
                    "line": node.lineno,
                    "file": self.current_file,
                    "args": len(node.args),
                }
            )

        # Check for os.path.join calls
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and node.func.value.attr == "path"
            and node.func.attr == "join"
        ):
            self.file_operations.append(
                {
                    "type": "os.path.join",
                    "line": node.lineno,
                    "file": self.current_file,
                    "args": len(node.args),
                }
            )

        # Check for Path operations
        elif isinstance(node.func, ast.Name) and node.func.id == "Path":
            self.file_operations.append(
                {
                    "type": "Path",
                    "line": node.lineno,
                    "file": self.current_file,
                    "args": len(node.args),
                }
            )

        self.generic_visit(node)


def audit_file_operations(directory: str) -> List[Tuple[str, List[dict]]]:
    """Audit file operations in Python files."""
    results = []

    for root, dirs, files in os.walk(directory):
        # Skip test directories and __pycache__
        dirs[:] = [
            d for d in dirs if not d.startswith(("__pycache__", ".git", "tests"))
        ]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    tree = ast.parse(content)
                    visitor = FileOperationVisitor()
                    visitor.current_file = file_path
                    visitor.visit(tree)

                    if visitor.file_operations:
                        results.append((file_path, visitor.file_operations))

                except Exception as e:
                    logger.warning(f"Could not parse {file_path}: {e}")

    return results


def generate_audit_report(directory: str) -> str:
    """Generate a file operations audit report."""
    operations = audit_file_operations(directory)

    report = ["File Operations Audit Report", "=" * 40, ""]

    if not operations:
        report.append("No file operations found.")
        return "\n".join(report)

    total_ops = 0
    for file_path, ops in operations:
        report.append(f"File: {file_path}")
        report.append("-" * len(f"File: {file_path}"))

        for op in ops:
            report.append(f"  Line {op['line']}: {op['type']} ({op['args']} args)")
            total_ops += 1

        report.append("")

    report.append(f"Total file operations found: {total_ops}")
    report.append(f"Files with operations: {len(operations)}")

    return "\n".join(report)


if __name__ == "__main__":
    # Generate audit report for the app directory
    app_dir = os.path.join(os.path.dirname(__file__), "..")
    report = generate_audit_report(app_dir)
    print(report)
