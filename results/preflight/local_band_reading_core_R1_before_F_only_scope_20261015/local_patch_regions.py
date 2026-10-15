"""Fixed patch-index groups shared across spatial frequency bands."""


def patch_regions(height, width, grid):
    assert grid in (1, 2) and height % grid == width % grid == 0
    rows, columns = height // grid, width // grid
    return tuple(
        tuple(y * width + x
              for y in range(row * rows, (row + 1) * rows)
              for x in range(column * columns, (column + 1) * columns))
        for row in range(grid) for column in range(grid)
    )
