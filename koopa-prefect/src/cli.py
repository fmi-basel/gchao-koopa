"""Basic CLI for running/config creation."""

import argparse
import sys

import koopa

import flows


def main():
    """Entrypoint function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--create-config", action="store_true")
    args = parser.parse_args()

    if args.create_config:
        config = koopa.config.create_default_config()
        koopa.io.save_config("./koopa.cfg", config)
        sys.exit(0)

    flows.workflow(args.config)


if __name__ == "__main__":
    main()
