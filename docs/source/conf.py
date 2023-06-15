import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'segment-lidar'
copyright = '2023, Anass Yarroudh'
author = 'Anass Yarroudh'
version = '0.1.5'
author_website = 'http://geomatics.ulg.ac.be/'
company = 'Geomatics Unit of ULiège'
github_url = 'https://github.com/Yarroudh/segment-lidar'
show_powered_by = False

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_context = {
    "display_github": True,
    "company": "Geomatics Unit of ULiège",
    "website": "https://github.com/Yarroudh/segment-lidar",
    'display_version': True,
    'versions': ['latest'],
    'current_version': 'latest',
    'version_dropdown': True,
    'display_github': True,
    'github_user': 'yarroudh',
    'github_repo': 'segment-lidar',
    'github_version': 'main/docs/',
}

html_show_sphinx = False