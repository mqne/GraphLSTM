from sys import stdout
from tqdm import tqdm


class TQDMHelper:
    """For printing additional info while a tqdm bar is active.

    Example:
        t = TQDMHelper()
        t.start()  # create newline to not overwrite previous output
        for x in range(2):
            g = tqdm_decorated_generator()
            for i in g:
                t.write("Status message for %i" % i)  # write/update message
                sleep(.05)
        t.stop()  # clear line
    """
    def __init__(self):
        from tqdm._utils import _term_move_up, _environ_cols_wrapper
        self._r_prefix = _term_move_up() + '\r'
        self._dynamic_ncols = _environ_cols_wrapper()

    def _write_raw(self, message):
        tqdm.write(self._r_prefix + message)

    def _clear(self):
        if self._dynamic_ncols:
            ncols = self._dynamic_ncols(stdout)
        else:
            ncols = 20
        self._write_raw(" " * ncols)

    @staticmethod
    def start():
        print()

    def write(self, message):
        self._clear()
        self._write_raw(message)

    def stop(self):
        self._clear()