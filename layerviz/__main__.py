"""
Run layerviz from the command line with the capture, compare, and serve subcommands.

Purpose:

- Capture one model from a factory spec, a checkpoint, and a sample into a bundle.
- Compare a model against a second checkpoint or a randomly initialized copy.
- Serve a bundle folder over http on 127.0.0.1.

Special Notes:

- Torch and the capture modules load only inside the capture and compare handlers, so
  python -m layerviz --help and serve stay fast and never import spconv.
"""

import argparse
import functools
import http.server
import sys
from pathlib import Path
from typing import Optional

RANDOM = "random"


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model, sample, and capture option flags shared by capture and compare."""
    parser.add_argument("--model", required=True, help="factory spec, such as pkg.mod:build")
    parser.add_argument("--checkpoint", help="checkpoint file loaded into the model")
    parser.add_argument("--state-key", help="checkpoint key searched first for the state dict")
    parser.add_argument("--sample", required=True, help="spec, or a .pt, .npy, or .npz file")
    parser.add_argument("--out", required=True, type=Path, help="bundle folder to write")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--include", nargs="+", help="module name patterns to keep")
    parser.add_argument("--exclude", nargs="+", help="module name patterns to drop")
    parser.add_argument("--layer-mb", type=float, default=16)
    parser.add_argument("--total-mb", type=float, default=1024)
    parser.add_argument("--inline-mb", type=float, default=16,
                        help="largest index.html written as one self contained page")
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser with the capture, compare, and serve subcommands.

    Returns:
        The parser. Each subcommand stores its handler as the handler attribute.
    """
    parser = argparse.ArgumentParser(prog="python -m layerviz")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="capture one model into a bundle")
    _add_model_arguments(capture)
    capture.set_defaults(handler=run_capture)
    compare = commands.add_parser("compare", help="compare two models in one bundle")
    _add_model_arguments(compare)
    compare.add_argument("--against", required=True,
                         help=f"checkpoint of the second model, or {RANDOM}")
    compare.add_argument("--labels", nargs=2, default=["A", "B"])
    compare.set_defaults(handler=run_compare)
    serve = commands.add_parser("serve", help="serve a bundle folder over http")
    serve.add_argument("folder", type=Path)
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=run_serve)
    return parser


def _capture_fields(args: argparse.Namespace) -> dict:
    """Return the CaptureOptions fields the model flags describe."""
    return {
        "sample_index": args.sample_index, "include": args.include, "exclude": args.exclude,
        "layer_mb": args.layer_mb, "total_mb": args.total_mb, "seed": args.seed,
    }


def run_capture(args: argparse.Namespace) -> int:
    """
    Capture the model the flags describe into a bundle and print its index.html path.

    Args:
        args: Parsed capture flags.

    Returns:
        The exit code 0.
    """
    from layerviz.api import capture
    from layerviz.loading import load_model, load_sample

    model = load_model(args.model, args.checkpoint, args.state_key, args.seed)
    batch = load_sample(args.sample)
    print(capture(model, batch, args.out, inline_mb=args.inline_mb, **_capture_fields(args)))
    return 0


def run_compare(args: argparse.Namespace) -> int:
    """
    Compare the model against a checkpoint or a random copy and print its index.html path.

    Args:
        args: Parsed compare flags.

    Returns:
        The exit code 0.

    Notes:
        The random copy is built with seed + 1 so its weights differ from the seeded model.
    """
    from layerviz.comparison import CompareOptions, compare
    from layerviz.loading import load_model, load_sample
    from layerviz.pipeline import CaptureOptions

    model_a = load_model(args.model, args.checkpoint, args.state_key, args.seed)
    if args.against == RANDOM:
        model_b = load_model(args.model, seed=args.seed + 1)
    else:
        model_b = load_model(args.model, args.against, args.state_key, args.seed)
    capture = CaptureOptions(**_capture_fields(args))
    options = CompareOptions(tuple(args.labels), capture, args.inline_mb)
    print(compare(model_a, model_b, load_sample(args.sample), args.out, options))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """
    Serve a bundle folder on 127.0.0.1 until interrupted, printing its URL first.

    Args:
        args: Parsed serve flags. Port 0 picks a free port.

    Returns:
        The exit code 0 after an interrupt.

    Raises:
        FileNotFoundError: When the folder does not exist.
    """
    if not args.folder.is_dir():
        raise FileNotFoundError(f"bundle folder {args.folder} does not exist")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.folder))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"http://127.0.0.1:{server.server_address[1]}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse argv and run the chosen subcommand.

    Args:
        argv: Command line arguments without the program name, sys.argv[1:] when None.

    Returns:
        The subcommand's exit code.
    """
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
