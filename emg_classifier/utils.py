import pathlib
import re
import typing as t

FILENAME_LABEL = re.compile(r"measurement_(.+?)(?:_\d+)?\.txt")


def load_from_sample(path: t.Union[pathlib.Path, str]):
    if isinstance(path, str):
        path = pathlib.Path(path)
        path.resolve(strict=True)

    for line in map(str.strip, path.read_text().split("\n")):
        if not line:
            continue
        yield int(line[29:])


def label_from_filename(filename: str):
    match = FILENAME_LABEL.search(filename)

    if match:
        return match.group(1)

    return filename
