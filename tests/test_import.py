from importlib.metadata import version

import mopt


def test_version_matches_installed_metadata():
    # hatch-vcs sets the distribution version from the git tag; __version__
    # must report that rather than a hardcoded string that can drift
    assert mopt.__version__ == version("mopt")


def test_version_is_not_the_uninstalled_placeholder():
    assert mopt.__version__ != "0.0.0+unknown"
