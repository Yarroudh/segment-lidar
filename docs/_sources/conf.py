"""Sphinx configuration for the published GitHub Pages documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "segment-lidar"
author = "Anass Yarroudh"
copyright = "2023–2026, Anass Yarroudh"
version = "0.3"
release = "0.3 series"

extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme"]
autodoc_typehints = "description"
templates_path = []
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_title = "segment-lidar"
html_copy_source = False
html_static_path = []
html_css_files = ["site.css"]
html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "style_external_links": True,
}


def _trim_generated_html(app, exception):
    """Keep committed HTML free of whitespace from theme templates."""
    if exception is not None or app.builder.name != "html":
        return
    for page in Path(app.outdir).glob("*.html"):
        lines = page.read_text(encoding="utf-8").splitlines()
        page.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def setup(app):
    app.connect("build-finished", _trim_generated_html)
