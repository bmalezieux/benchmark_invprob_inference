# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ------------------------------------------------------
import importlib.util
import sys
import types
from pathlib import Path

# Add project root to path so Sphinx can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _load_pkg_modules(package_name: str, package_dir: Path):
    """Load a folder as a package so autosummary can import submodules."""
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(package_dir)]
        first_py = next(package_dir.glob('*.py'), None)
        pkg.__file__ = str(first_py) if first_py else None
        sys.modules[package_name] = pkg

    for module_path in package_dir.glob('*.py'):
        if module_path.name.startswith('_'):
            continue
        module_name = f"{package_name}.{module_path.stem}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)


# Expose datasets/ and solvers/ as packages for autodoc/autosummary
_load_pkg_modules('datasets', project_root / 'datasets')
_load_pkg_modules('solvers', project_root / 'solvers')


def _write_module_stub(package: str, module_stem: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    title = f"{package}.{module_stem}"
    underline = '=' * len(title)
    content = f"""
{title}
{underline}

.. automodule:: {package}.{module_stem}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    (target_dir / f"{module_stem}.rst").write_text(content)


def _generate_api_stubs(app):
    generated_base = Path(__file__).parent / 'api' / 'generated'
    for package in ('datasets', 'solvers'):
        pkg_dir = project_root / package
        target_dir = generated_base / package
        for old_stub in target_dir.glob('*.rst'):
            old_stub.unlink()
        for module_path in pkg_dir.glob('*.py'):
            if module_path.name.startswith('_'):
                continue
            _write_module_stub(package, module_path.stem, target_dir)


def setup(app):
    app.connect('builder-inited', _generate_api_stubs)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Benchmark Inverse Problem Inference'
copyright = '2026, authors'
author = 'authors'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_gallery.gen_gallery',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# Autosummary: generate stub pages so new modules under datasets/ or solvers/
# automatically appear in the API without manual edits.
autosummary_generate = True
autosummary_imported_members = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom.css')
    app.connect('builder-inited', _generate_api_stubs)

html_theme_options = {
    'logo_only': False,
    'collapse_navigation': False,
}

# Autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Sphinx-Gallery settings
sphinx_gallery_conf = {
    'examples_dirs': ['../../examples'],
    'gallery_dirs': ['auto_examples'],
    'doc_module': ('deepinv', 'torch','benchopt'),
    'reference_url': {},
    'abort_on_example_error': False,
    'filename_pattern': '/demo_',  # Execute demo_*.py files
    'plot_gallery': True,  # Enable execution
    'download_all_examples': False,  # Don't create zip files
}

# Intersphinx mapping
intersphinx_mapping = {
     'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
