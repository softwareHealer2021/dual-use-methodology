import re
from pathlib import Path


def safe_name(name: str) -> str:
    """
    Convert names into filesystem-safe strings.

    Example:
    "Random Forest" -> "random_forest"
    "Header Features" -> "header_features"
    """

    name = name.lower().strip()

    name = re.sub(r"[^\w\s-]", "", name)

    name = re.sub(r"[\s\-]+", "_", name)

    return name



def make_dir(directory):
    """
    Create directory if it does not exist.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory