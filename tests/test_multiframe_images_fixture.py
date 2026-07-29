from pathlib import Path

from PIL import Image


def test_multiframe_images_have_valid_varkoder_names_and_metadata(multiframe_images):
    df, labels, levels = multiframe_images

    for row_path in df["path"]:
        p = Path(row_path)
        assert p.name.endswith("+cgr+k6.apng")
        assert "@stack+" in p.name

        with Image.open(p) as im:
            assert im.info.get("varkoderMapping") == "cgr"
            assert im.info.get("varkoderFrameSizes") == "3000000,1000000,300000"
            assert im.info.get("varkoderFormatVersion") == "2"
            assert im.info.get("varkoderKeywords") in labels
