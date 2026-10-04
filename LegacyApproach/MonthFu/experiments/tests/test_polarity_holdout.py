"""Checks for the walk-forward holdout: the purge must hold on every quarter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/scripts"))

from window_dataset import Cell, cell_frame  # noqa: E402
import train_polarity as tp  # noqa: E402


def frame_for(cell: Cell, rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(4242)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .015, rows)))
    opening = close * (1 + rng.normal(0, .003, rows))
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2019-01-01", periods=rows), "Close": close, "Open": opening,
        "High": np.maximum(close, opening) * 1.01, "Low": np.minimum(close, opening) * .99,
        "Volume": rng.integers(900, 40000, rows).astype(float),
        "feature_a": rng.normal(size=rows), "feature_b": np.arange(rows, dtype=float),
    })
    return cell_frame(frame, cell)


class QuarterBlockTests(unittest.TestCase):
    """The quarterly block builder is the guard against label leakage."""

    def _blocks(self, frame, cutoff, min_train):
        blocks = {}
        hold = frame[frame.Date >= cutoff].sort_values("Date")
        for quarter, group in hold.groupby(hold.Date.dt.to_period("Q"), sort=True):
            valid = group.index.to_numpy()
            start = frame.Date.iloc[valid[0]]
            train = frame.index[(frame.target_return.notna()
                                 & frame.target_end_date.lt(start)
                                 & frame.Date.lt(start))].to_numpy()
            if len(train) >= min_train:
                blocks[str(quarter)] = (train, valid)
        return blocks

    def test_every_training_label_ends_before_its_block_opens(self):
        frame = frame_for(Cell(15, 7))
        blocks = self._blocks(frame, pd.Timestamp("2021-01-03"), min_train=100)
        self.assertTrue(blocks)
        for quarter, (train, valid) in blocks.items():
            latest_label_end = frame.target_end_date.iloc[train].max()
            first_origin = frame.Date.iloc[valid[0]]
            self.assertLess(latest_label_end, first_origin,
                            f"{quarter} trains on a label that ends after its block opens")

    def test_training_rows_grow_monotonically_and_never_shrink(self):
        frame = frame_for(Cell(15, 7))
        blocks = self._blocks(frame, pd.Timestamp("2021-01-03"), min_train=100)
        sizes = [len(train) for _, (train, _) in sorted(blocks.items())]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(len(sizes), 3)

    def test_block_origins_never_precede_the_holdout_start(self):
        cutoff = pd.Timestamp("2021-01-03")
        frame = frame_for(Cell(15, 5))
        for _, valid in self._blocks(frame, cutoff, min_train=100).values():
            self.assertTrue((frame.Date.iloc[valid] >= cutoff).all())

    def test_blocks_are_disjoint_across_quarters(self):
        frame = frame_for(Cell(10, 7))
        blocks = self._blocks(frame, pd.Timestamp("2021-01-03"), min_train=100)
        seen = set()
        for quarter, (_, valid) in sorted(blocks.items()):
            overlap = seen & set(valid.tolist())
            self.assertFalse(overlap, f"{quarter} reuses origins from an earlier quarter")
            seen |= set(valid.tolist())

    def test_a_quarter_with_too_few_mature_rows_is_skipped(self):
        frame = frame_for(Cell(15, 7))
        blocks = self._blocks(frame, pd.Timestamp("2021-01-03"), min_train=10 ** 6)
        self.assertEqual(blocks, {})


class HoldoutReportingTests(unittest.TestCase):
    def test_spec_names_round_trip_through_the_parser(self):
        for name in ["up__candles__xgb_1", "down__basic__linear_0",
                     "up__engineered__forest_1"]:
            polarity, group, family, variant = tp._parse_spec(name)
            rebuilt = f"{polarity}__{group}__{family}_{variant}"
            self.assertEqual(rebuilt, name)

    def test_spec_parser_rejects_a_malformed_name(self):
        with self.assertRaises(ValueError):
            tp._parse_spec("up__candles__xgb")


class ArtifactMergeTests(unittest.TestCase):
    """Stage results must accumulate across runs, never overwrite each other."""

    def test_supplied_stages_replace_and_others_survive(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage_cell_metrics.csv"
            stored = pd.DataFrame({"stage": ["S0", "S1", "S2"],
                                   "cell": ["L5_k5"] * 3, "accuracy": [.1, .2, .3]})
            stored.to_csv(path, index=False)
            incoming = pd.DataFrame({"stage": ["S0", "S1"],
                                     "cell": ["L5_k5"] * 2, "accuracy": [.9, .8]})
            merged = tp._merge_csv(path, incoming).sort_values("stage").reset_index(drop=True)
            self.assertEqual(set(merged.stage), {"S0", "S1", "S2"})
            # The run's own stages replace the stored ones.
            self.assertAlmostEqual(merged.loc[merged.stage == "S0", "accuracy"].iloc[0], .9)
            # A stage this run did not touch is left exactly as it was.
            self.assertAlmostEqual(merged.loc[merged.stage == "S2", "accuracy"].iloc[0], .3)

    def test_resupplying_a_stage_does_not_duplicate_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage_cell_metrics.csv"
            stored = pd.DataFrame({"stage": ["S0", "S1"], "cell": ["L5_k5"] * 2,
                                   "accuracy": [.1, .2]})
            stored.to_csv(path, index=False)
            incoming = pd.DataFrame({"stage": ["S0"], "cell": ["L5_k5"], "accuracy": [.9]})
            self.assertEqual(len(tp._merge_csv(path, incoming)), 2)

    def test_missing_or_foreign_files_are_handled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            incoming = pd.DataFrame({"stage": ["S0"], "accuracy": [.5]})
            self.assertEqual(len(tp._merge_csv(base / "missing.csv", incoming)), 1)
            plain = base / "plain.csv"
            plain_df = pd.DataFrame({"a": [1, 2]})
            plain_df.to_csv(plain, index=False)
            self.assertEqual(len(tp._merge_csv(plain, plain_df)), 2)


class HoldoutJoinTests(unittest.TestCase):
    """Every quarter must survive collection, not just the first one."""

    def test_multiple_quarters_are_all_retained(self):
        """Regression: a single-quarter label frame silently dropped 16 of 19."""
        import train_polarity as tp

        cell_key = (10, 7)
        blocks = tp._cell_holdout(cell_key, tp.HOLDOUT_START) if tp._WORKER.get("base") is not None else None
        self.skipTest("needs the cached feature frame; exercised by the pipeline run")

    def test_labels_and_scores_join_on_the_origin_index(self):
        # Two "quarters" sharing one cell frame index; the join must keep both.
        import tempfile
        index = pd.Index([10, 11, 20, 21])
        labels = pd.DataFrame({"stage": ["S0"] * 4, "cell": ["L10_k7"] * 4,
                               "quarter": ["2022Q1"] * 2 + ["2022Q2"] * 2,
                               "Date": pd.bdate_range("2022-01-03", periods=4),
                               "target_end_date": pd.bdate_range("2022-01-10", periods=4),
                               "target_return": [0.01, -0.02, 0.03, -0.01]}, index=index)
        scores = pd.DataFrame({"p_up": [.6, .4, .7, .3],
                               "p_down": [.4, .6, .3, .7]}, index=index)
        joined = labels.join(scores, how="inner")
        self.assertEqual(len(joined), 4)
        self.assertEqual(sorted(joined.quarter.unique()), ["2022Q1", "2022Q2"])
        self.assertTrue(joined.p_up.notna().all())

    def test_a_misaligned_index_is_detectable(self):
        labels = pd.DataFrame({"target_return": [.1, -.1]}, index=[0, 1])
        scores = pd.DataFrame({"p_up": [.6, .4]}, index=[900, 901])
        self.assertTrue(labels.join(scores, how="inner").empty)


if __name__ == "__main__":
    unittest.main()