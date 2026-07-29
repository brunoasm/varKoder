import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat
from fastai.vision.all import (
    CategoryBlock, ColReader, ColSplitter, DataBlock, ImageBlock, MultiCategoryBlock,
    Resize, ResizeMethod, vision_learner,
)

from varKoder.commands.query import QueryCommand
from varKoder.core.utils import format_bp_human_readable


def _save_single_frame(path, **text_items):
    info = PngInfo()
    for k, v in text_items.items():
        info.add_text(k, v)
    Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(path, pnginfo=info)


def _save_multiframe(path, levels, **text_items):
    info = PngInfo()
    for k, v in text_items.items():
        info.add_text(k, v)
    frames = [Image.fromarray(np.full((8, 8), lv, dtype="uint8")).convert("L") for lv in levels]
    frames[0].save(path, save_all=True, append_images=frames[1:], pnginfo=info)


def _make_query_command(tmp_path, all_frames=False):
    cmd = QueryCommand.__new__(QueryCommand)
    cmd.args = argparse.Namespace(all_frames=all_frames)
    cmd.inter_dir = tmp_path / "inter"
    return cmd


def test_expand_query_items_default_one_item_per_file(tmp_path):
    single_path = tmp_path / "sample1@00500K+cgr+k7.png"
    _save_single_frame(single_path, varkoderKeywords="alpha")

    multi_path = tmp_path / "sample2@stack+cgr+k7.apng"
    _save_multiframe(
        multi_path, [200, 130, 20],
        varkoderKeywords="beta", varkoderFrameSizes="3000000,1000000,300000",
    )

    cmd = _make_query_command(tmp_path, all_frames=False)
    items = cmd._expand_query_items([single_path, multi_path])

    assert len(items) == 2
    assert items[0]["report_path"] == single_path
    assert items[0]["loader_path"] == single_path
    assert items[0]["bp"] == 500000

    assert items[1]["report_path"] == multi_path
    assert items[1]["loader_path"] == multi_path
    assert items[1]["bp"] == 3000000  # representative = frame 0 = largest bp


def test_expand_query_items_all_frames_expands_multiframe_only(tmp_path):
    single_path = tmp_path / "sample1@00500K+cgr+k7.png"
    _save_single_frame(single_path, varkoderKeywords="alpha")

    multi_path = tmp_path / "sample2@stack+cgr+k7.apng"
    _save_multiframe(
        multi_path, [200, 130, 20],
        varkoderKeywords="beta", varkoderFrameSizes="3000000,1000000,300000",
    )

    cmd = _make_query_command(tmp_path, all_frames=True)
    items = cmd._expand_query_items([single_path, multi_path])

    # Single-frame file: unaffected, still exactly one item.
    single_items = [it for it in items if it["report_path"] == single_path]
    assert len(single_items) == 1
    assert single_items[0]["loader_path"] == single_path

    # Multiframe file: one item per frame, report_path stays the original
    # file for every expanded item, bp follows varkoderFrameSizes in order.
    multi_items = [it for it in items if it["report_path"] == multi_path]
    assert len(multi_items) == 3
    assert [it["bp"] for it in multi_items] == [3000000, 1000000, 300000]
    assert all(it["report_path"] == multi_path for it in multi_items)
    assert len({it["loader_path"] for it in multi_items}) == 3  # 3 distinct temp files
    for it in multi_items:
        assert it["loader_path"].parent == cmd.inter_dir / "all_frames_tmp"


def _make_query_ready_images(tmp_path, label_sets):
    rng = np.random.default_rng(0)
    rows = []
    for i, lab in enumerate(label_sets):
        arr = (rng.random((32, 32, 3)) * 255).astype("uint8")
        p = tmp_path / f"sample{i}@{format_bp_human_readable(500000 + i)}+cgr+k7.png"
        info = PngInfo()
        info.add_text("varkoderKeywords", lab)
        info.add_text("varkoderLowQualityFlag", "False")
        info.add_text("varkoderBaseFreqSd", "0.01")
        info.add_text("varkoderMapping", "cgr")
        Image.fromarray(arr).save(p, pnginfo=info)
        rows.append({"path": str(p), "labels": lab, "is_valid": i >= len(label_sets) - 1})
    return pd.DataFrame(rows)


def _tiny_multilabel_learner(df, vocab):
    dbl = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab, encoded=False)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels", label_delim=";"),
        item_tfms=Resize(32, method=ResizeMethod.Squish),
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    return vision_learner(
        dls, "resnet18", pretrained=False, normalize=True,
        loss_func=BCEWithLogitsLossFlat(),
    )


def _run_query(tmp_path, learn, is_multilabel, threshold=0.7, include_probs=False):
    outdir = tmp_path / "out"
    cmd = QueryCommand.__new__(QueryCommand)
    cmd.args = argparse.Namespace(
        images=True, input=str(tmp_path), outdir=str(outdir), model="unused",
        threshold=threshold, include_probs=include_probs, max_batch_size=2,
        int_folder=None, keep_images=False, all_frames=False, overwrite=True,
    )
    cmd.np_rng = np.random.default_rng(0)
    cmd.all_stats = {}
    cmd.inter_dir = Path(tempfile.mkdtemp(prefix="barcoding_"))
    cmd.images_d = tmp_path
    cmd.is_multilabel = is_multilabel
    cmd.load_model = lambda n: learn
    cmd.run()
    return pd.read_csv(outdir / "predictions.csv")


def test_query_multilabel_output_columns_and_threshold_boundary(tmp_path):
    df = _make_query_ready_images(
        tmp_path, ["alpha", "beta", "alpha;beta", "beta"]
    )
    vocab = ["alpha", "beta"]
    learn = _tiny_multilabel_learner(df, vocab)

    # Row 0: exactly at threshold (boundary is inclusive, >=).
    # Row 1: just below threshold (excluded).
    # Row 2: both above. Row 3: both below.
    fake_pp = torch.tensor([
        [0.7, 0.0],
        [0.69, 0.0],
        [0.9, 0.8],
        [0.1, 0.2],
    ])
    learn.get_preds = lambda **kwargs: (fake_pp, None)

    out_df = _run_query(tmp_path, learn, is_multilabel=True, threshold=0.7)

    assert list(out_df.columns) == [
        "varKode_image_path", "sample_id", "query_basepairs", "query_kmer_len",
        "query_mapping", "trained_model_path", "actual_labels",
        "possible_low_quality", "basefrequency_sd", "prediction_type",
        "prediction_threshold", "predicted_labels",
    ]
    assert "best_pred_label" not in out_df.columns
    assert "best_pred_prob" not in out_df.columns

    assert out_df["predicted_labels"].where(pd.notna(out_df["predicted_labels"]), None).tolist() == ["alpha", None, "alpha;beta", None]


def _tiny_single_label_learner(df, vocab):
    dbl = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels"),
        item_tfms=Resize(32, method=ResizeMethod.Squish),
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    return vision_learner(
        dls, "resnet18", pretrained=False, normalize=True,
        loss_func=CrossEntropyLossFlat(),
    )


def test_query_single_label_output_columns_and_best_pred(tmp_path):
    df = _make_query_ready_images(tmp_path, ["alpha", "beta", "alpha", "beta"])
    vocab = ["alpha", "beta"]
    learn = _tiny_single_label_learner(df, vocab)

    fake_pp = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.55, 0.45],
        [0.5, 0.5],
    ])
    learn.get_preds = lambda **kwargs: (fake_pp, None)

    out_df = _run_query(tmp_path, learn, is_multilabel=False)

    assert list(out_df.columns) == [
        "varKode_image_path", "sample_id", "query_basepairs", "query_kmer_len",
        "query_mapping", "trained_model_path", "actual_labels",
        "possible_low_quality", "basefrequency_sd", "prediction_type",
        "best_pred_label", "best_pred_prob",
    ]
    assert "prediction_threshold" not in out_df.columns
    assert "predicted_labels" not in out_df.columns

    assert out_df["best_pred_label"].tolist() == ["alpha", "beta", "alpha", "alpha"]
    assert out_df["best_pred_prob"].tolist() == pytest.approx([0.9, 0.8, 0.55, 0.5])
