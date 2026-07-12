from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.casino_dataset import StrategyDataCollator
from src.config import TrainConfig
from src.losses import align_labels_and_response_mask, response_nll_stats
from src.strategy_labels import CANONICAL_LABELS, assert_exact_label_order, load_canonical_labels, validate_observed_labels


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        values = [10 + (ord(char) % 20) for char in text if not char.isspace()]
        return {"input_ids": values or []}


class AlignmentTests(unittest.TestCase):
    def aligned(self, labels, mask, prefix=0, vocab=64):
        length = labels.size(1) + prefix
        logits = torch.zeros(labels.size(0), length, vocab)
        aligned = align_labels_and_response_mask(
            labels, mask, num_virtual_tokens=prefix, logits_sequence_length=length
        )
        return logits, aligned

    def test_no_padding(self):
        labels = torch.tensor([[-100, 5, 6]])
        mask = torch.tensor([[0, 1, 1]])
        logits, (labels, mask) = self.aligned(labels, mask)
        _, counts = response_nll_stats(logits, labels, mask)
        self.assertEqual(counts.tolist(), [2])

    def test_right_padding(self):
        labels = torch.tensor([[-100, 5, 6, -100]])
        mask = torch.tensor([[0, 1, 1, 0]])
        logits, (labels, mask) = self.aligned(labels, mask)
        _, counts = response_nll_stats(logits, labels, mask)
        self.assertEqual(counts.tolist(), [2])

    def test_chosen_rejected_different_lengths(self):
        chosen = torch.tensor([[-100, 5, 6, 7]])
        rejected = torch.tensor([[-100, 8, -100, -100]])
        for labels, mask, expected in (
            (chosen, chosen.ne(-100), 3), (rejected, rejected.ne(-100), 1)
        ):
            logits, aligned = self.aligned(labels, mask)
            _, counts = response_nll_stats(logits, *aligned)
            self.assertEqual(counts.tolist(), [expected])

    def test_one_response_token(self):
        labels = torch.tensor([[-100, 5]])
        logits, aligned = self.aligned(labels, labels.ne(-100))
        _, counts = response_nll_stats(logits, *aligned)
        self.assertEqual(counts.tolist(), [1])

    def test_response_plus_eos(self):
        cfg = TrainConfig(max_length=32)
        collator = StrategyDataCollator(FakeTokenizer(), cfg)
        item = {"dialogue_id": 1, "turn_index": 1, "speaker_id": "a", "primary_strategy": "self-need",
                "strategy_id": 4, "all_strategies": ["self-need"], "prompt": "P", "target": "R"}
        batch = collator([item])
        self.assertEqual(int(batch["response_mask"].sum()), 6)  # R + literal <eos> in fake tokenizer
        self.assertTrue(torch.equal(batch["labels"].ne(-100), batch["response_mask"]))

    def test_prefix_virtual_tokens(self):
        labels = torch.tensor([[-100, 5, 6]])
        logits, (aligned_labels, aligned_mask) = self.aligned(labels, labels.ne(-100), prefix=3)
        self.assertTrue(torch.all(aligned_labels[:, :3] == -100))
        self.assertFalse(torch.any(aligned_mask[:, :3]))
        _, counts = response_nll_stats(logits, aligned_labels, aligned_mask)
        self.assertEqual(counts.tolist(), [2])

    def test_empty_response_fails(self):
        labels = torch.tensor([[-100, -100]])
        logits, aligned = self.aligned(labels, labels.ne(-100))
        with self.assertRaises(ValueError):
            response_nll_stats(logits, *aligned)

    def test_all_ignore_labels_fail(self):
        labels = torch.full((2, 3), -100)
        logits, aligned = self.aligned(labels, labels.ne(-100))
        with self.assertRaises(ValueError):
            response_nll_stats(logits, *aligned)

    def test_shape_mismatch_fails(self):
        labels = torch.tensor([[-100, 5]])
        with self.assertRaises(ValueError):
            align_labels_and_response_mask(
                labels, torch.tensor([[0]]), num_virtual_tokens=0, logits_sequence_length=2
            )
        with self.assertRaises(ValueError):
            align_labels_and_response_mask(
                labels, labels.ne(-100), num_virtual_tokens=0, logits_sequence_length=3
            )

    def test_nan_fails(self):
        labels = torch.tensor([[-100, 5]])
        logits, aligned = self.aligned(labels, labels.ne(-100))
        logits[:] = float("nan")
        with self.assertRaises(FloatingPointError):
            response_nll_stats(logits, *aligned)


class LabelSpaceTests(unittest.TestCase):
    def test_canonical_file(self):
        self.assertEqual(load_canonical_labels("configs/strategy_label_space.json"), list(CANONICAL_LABELS))

    def test_unknown_and_missing_fail(self):
        with self.assertRaises(ValueError):
            validate_observed_labels(["unknown"], CANONICAL_LABELS, require_all=False, source="test")
        with self.assertRaises(ValueError):
            validate_observed_labels([CANONICAL_LABELS[0]], CANONICAL_LABELS, require_all=True, source="test")

    def test_checkpoint_order_mismatch_fails(self):
        with self.assertRaises(ValueError):
            assert_exact_label_order(list(reversed(CANONICAL_LABELS)), CANONICAL_LABELS, source="checkpoint")


if __name__ == "__main__":
    unittest.main()
