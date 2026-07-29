import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from varKoder.commands.query import QueryCommand


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
