import tempfile
import unittest
from pathlib import Path

from scripts.d3_stage1a_readiness import artifact_audit, require_audit_pass


class Direction3Stage1AReadinessTests(unittest.TestCase):
    def test_readiness_audit_fails_when_required_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            root.mkdir()
            audit = artifact_audit(root)
            with self.assertRaises(AssertionError):
                require_audit_pass(audit)


if __name__ == "__main__":
    unittest.main()
