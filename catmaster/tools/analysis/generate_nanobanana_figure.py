"""Compatibility imports for the renamed general figure-generation tool."""

from catmaster.tools.analysis.generate_figure import (
    GenerateFigureInput as GenerateNanoBananaFigureInput,
    generate_figure as generate_nanobanana_figure,
)

__all__ = ["GenerateNanoBananaFigureInput", "generate_nanobanana_figure"]
