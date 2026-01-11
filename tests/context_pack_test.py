#!/usr/bin/env python3
from catmaster.runtime.context_pack import _parse_key_files


def main() -> None:
    sample = """
### Key Files
- FILE[core]: data/input.txt | core input
  FILE[aux]: outputs/result.json | generated
    - FILE[spaced]:  logs/run.log  | trailing spaces
Not a file line
"""
    paths = _parse_key_files(sample)
    assert paths == [
        "data/input.txt",
        "outputs/result.json",
        "logs/run.log",
    ], f"Unexpected key files: {paths}"

    sample_multi = "FILE[a]: one.txt | a\nFILE[b]: two.txt | b\n"
    paths_multi = _parse_key_files(sample_multi)
    assert paths_multi == ["one.txt", "two.txt"], f"Unexpected multi paths: {paths_multi}"


if __name__ == "__main__":
    main()
