"""Cheap *risk* pre-filter used before OCR in the timeline screen path.

Generic text is deliberately not a risk signal.  The detector only returns
compact QR-like regions and overlay-shaped regions near the frame boundary.
The caller must still confirm those regions (OCR pattern or temporal stability)
before publishing a risk candidate.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


def _connected_components(active: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(active, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    rows, columns = active.shape
    for row in range(rows):
        for column in range(columns):
            if not active[row, column] or visited[row, column]:
                continue
            queue = deque([(row, column)])
            visited[row, column] = True
            component: list[tuple[int, int]] = []
            while queue:
                current_row, current_column = queue.popleft()
                component.append((current_row, current_column))
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        if row_delta == 0 and column_delta == 0:
                            continue
                        next_row = current_row + row_delta
                        next_column = current_column + column_delta
                        if (
                            0 <= next_row < rows
                            and 0 <= next_column < columns
                            and active[next_row, next_column]
                            and not visited[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            queue.append((next_row, next_column))
            components.append(component)
    return components


def detect_candidate_regions(
    image: Image.Image,
    *,
    maximum_regions: int = 8,
) -> list[dict[str, Any]]:
    """Return OCR pre-filter boxes, never a final risk decision.

    Ordinary scene text and the usual bottom-centre subtitle band are excluded.
    This avoids the former failure mode where every subtitle frame was treated
    as a risk hit and expanded to a 0.5-second full-timeline scan.
    """

    original_width, original_height = image.size
    maximum_side = max(original_width, original_height)
    scale = min(1.0, 640.0 / float(maximum_side))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    gray = np.asarray(image.resize((width, height)).convert("L"), dtype=np.int16)
    horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    vertical = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = (horizontal + vertical) >= 52

    cell = 24
    rows = (height + cell - 1) // cell
    columns = (width + cell - 1) // cell
    densities = np.zeros((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            tile = edges[
                row * cell : min(height, (row + 1) * cell),
                column * cell : min(width, (column + 1) * cell),
            ]
            densities[row, column] = float(tile.mean()) if tile.size else 0.0

    # Dilation by one grid cell lets neighbouring glyphs form one OCR crop.
    active = densities >= 0.16
    expanded = active.copy()
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == 0 and column_delta == 0:
                continue
            shifted = np.zeros_like(active)
            source_rows = slice(max(0, -row_delta), min(rows, rows - row_delta))
            source_columns = slice(
                max(0, -column_delta), min(columns, columns - column_delta)
            )
            target_rows = slice(max(0, row_delta), min(rows, rows + row_delta))
            target_columns = slice(
                max(0, column_delta), min(columns, columns + column_delta)
            )
            shifted[target_rows, target_columns] = active[
                source_rows, source_columns
            ]
            expanded |= shifted

    candidates: list[dict[str, Any]] = []
    for component in _connected_components(expanded):
        component_rows = [value[0] for value in component]
        component_columns = [value[1] for value in component]
        top = min(component_rows) * cell
        bottom = min(height, (max(component_rows) + 1) * cell)
        left = min(component_columns) * cell
        right = min(width, (max(component_columns) + 1) * cell)
        box_width = right - left
        box_height = bottom - top
        source_density = float(
            densities[
                min(component_rows) : max(component_rows) + 1,
                min(component_columns) : max(component_columns) + 1,
            ].max()
        )
        horizontal_text = box_width >= box_height * 1.35 and box_width >= 48
        aspect = box_width / float(max(1, box_height))
        compact_dense = (
            source_density >= 0.32
            and 0.65 <= aspect <= 1.55
            and 32 <= box_width <= width * 0.38
            and 32 <= box_height <= height * 0.55
        )
        centre_x = (left + right) / 2.0 / float(width)
        centre_y = (top + bottom) / 2.0 / float(height)
        near_boundary = (
            centre_x <= 0.24
            or centre_x >= 0.76
            or centre_y <= 0.22
            or centre_y >= 0.86
        )
        ordinary_subtitle_band = 0.20 <= centre_x <= 0.80 and centre_y >= 0.62
        overlay_geometry = horizontal_text and near_boundary and not ordinary_subtitle_band
        if not compact_dense and not overlay_geometry:
            continue
        padding = 8
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        candidates.append(
            {
                "box": [
                    round(left / scale),
                    round(top / scale),
                    round(right / scale),
                    round(bottom / scale),
                ],
                "reason": "qr_like_geometry" if compact_dense else "overlay_text_geometry",
                "score": round(source_density, 6),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), item["box"]))
    return candidates[:maximum_regions]
