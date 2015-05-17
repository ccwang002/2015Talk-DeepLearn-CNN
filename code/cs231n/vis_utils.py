from math import sqrt, ceil
import numpy as np
from IPython.html import widgets
from IPython.display import Javascript


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                grid_slice_y = slice(y * H + y, (y + 1) * H + y)
                grid_slice_x = slice(x * W + x, (x + 1) * W + x)
                G[grid_slice_y, grid_slice_x, :] = Xs[n, :, :, :]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G

def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            grid_slice_y = slice(y * H + y, (y + 1) * H + y)
            grid_slice_x = slice(x * W + x, (x + 1) * W + x)
            G[grid_slice_y, grid_slice_x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


class ParametersInspectorWindow:
    instance = None

    def __init__(self, ipython, params):
        # singleton pattern
        if ParametersInspectorWindow.instance:
            raise Exception(
                "Only one instance of the Variable Inspector can exist "
                "at a time. Call close() on the active instance before "
                "creating a new instance. \n"
                "If you have lost the handle to the active instance, "
                "you can re-obtain it via "
                "`ParametersInspectorWindow.instance`."
            )

        ParametersInspectorWindow.instance = self
        self.closed = False
        self._value = params
        # construct widget
        self._box = widgets.Box()
        self._box._dom_classes = ['inspector']
        self._box.background_color = '#fff'
        self._box.border_color = '#ccc'
        self._box.border_width = 1
        self._box.border_radius = 5

        self._modal_body = widgets.VBox()
        self._modal_body.overflow_y = 'scroll'
        self._modal_body.padding = '0 10px 0'
        # self._modal_body.width = '300px'
        # self._modal_body.height = '600px'

        self._modal_body_label = widgets.HTML(value='Not hooked')
        self._modal_body.children = [self._modal_body_label]

        self._box.children = [
            self._modal_body,
        ]

        self._ipython = ipython
        self._ipython.events.register('post_run_cell', self._fill)

    def close(self):
        """Close and remove hooks."""
        if not self.closed:
            self._ipython.events.unregister('post_run_cell', self._fill)
            self._box.close()
            self.closed = True
            ParametersInspectorWindow.instance = None

    def _fill(self):
        """Fill self with variable information."""
        try:
            self._modal_body_label.value = (
                '<h4>Inspector</h4>'
                '<table class="table table-bordered table-striped">'
                '<tr><th>Parameter</th><th>Value</th></tr>'
                '<tr><td>' +
                '</td></tr><tr><td>'.join([
                    '{0}</td><td>{1}</td>'.format(k, v)
                    for k, v in self.value.items()
                ]) +
                '</td></tr>'
                '</table>'
            )
        except:
            self._modal_body_label.value = (
                '<h4>Inspector</h4>'
                '<p style="color: red"><b>ERROR</b> '
                'failed to obtain <code>self.value</code></p>'
            )

    def _ipython_display_(self):
        """Called when display() or pyout is used to display the Variable
        Inspector."""
        self._box._ipython_display_()

    def detach(self):
        return Javascript("""
            $('div.inspector')
            .detach()
            .prependTo($('body'))
            .css({
                'z-index': 999,
                position: 'fixed',
                width: 'auto',
                'box-shadow': '5px 5px 12px -3px black',
                opacity: 0.9
            })
            .draggable()
            .resizable();
        """)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._fill()
