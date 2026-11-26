"""Shared dashboard chart styling."""

import plotly.express as px

COLOR = "#137e73"
PALETTE = [COLOR, "#395a85", "#d9874a", "#7b669c"]
TEMPLATE = "plotly_white"


def bars(data, **kwargs):
    return px.bar(data, template=TEMPLATE, color_discrete_sequence=PALETTE, **kwargs)
