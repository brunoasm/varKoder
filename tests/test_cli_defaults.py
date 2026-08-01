from varKoder.cli import setup_parser
from varKoder.core.config import DEFAULT_MODEL


def test_train_defaults():
    # setup_parser() (cli.py:37) returns the configured ArgumentParser;
    # `train` takes positionals `input` and `outdir` (cli.py:178-181).
    args = setup_parser().parse_args(["train", "in", "out"])
    assert args.architecture == "vit_large_patch32_224"
    assert args.pretrained_model == DEFAULT_MODEL
