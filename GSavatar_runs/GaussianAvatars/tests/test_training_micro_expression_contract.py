import ast
import unittest
from pathlib import Path


class MicroExpressionTrainingContractTest(unittest.TestCase):
    def test_mesh_selection_is_not_disabled_in_offset_only_training(self):
        source = Path(__file__).parents[1] / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            branch_calls_select_mesh = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "select_mesh_by_timestep"
                for child in ast.walk(node)
            )
            if branch_calls_select_mesh:
                condition = ast.unparse(node.test)
                self.assertNotIn("micro_expression_only", condition)
                return

        self.fail("Training loop no longer selects a FLAME mesh timestep.")


if __name__ == "__main__":
    unittest.main()
