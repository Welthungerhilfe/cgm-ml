""" jupyter extension:

* shows starttime
* shows elapsedtime
* sounds alert on cell finish
* %s and %%s magic to suppress line/cell execution
"""
import time
from IPython.core.magics.execution import _format_time
from IPython.display import display as d
from IPython.display import Audio
from IPython.core.display import HTML
import numpy as np
import logging
log = logging.getLogger("cellevents")
log.setLevel(logging.INFO)

def alert():
    """ makes sound on client using javascript (works with remote server) """
    framerate = 44100
    duration = .05
    freq = 300
    t = np.linspace(0, duration, int(framerate * duration))
    data = np.sin(2 * np.pi * freq * t)
    d(Audio(data, rate=framerate, autoplay=True))
    hide_audio()

def hide_audio():
    """ hide the audio control """
    d(HTML("<style>audio{display:none}</style>"))

def s(line, cell=None):
    '''Skips execution of the current line/cell.'''
    pass

class Cell(object):
    """ action cell events """

    def __init__(self):
        self.start_time = None

    def pre_run_cell(self):
        log.info("starting")
        self.start_time = time.time()

    def post_run_cell(self):
        # show the elapsed time
        if self.start_time:
            diff = time.time() - self.start_time
            print('time: %s' % _format_time(diff))

            # alert finish if more than 30 seconds
            if diff > 30:
                alert()
        self.start_time = None


cell = Cell()

def load_ipython_extension(ip):
    hide_audio()
    ip.events.register('pre_run_cell', cell.pre_run_cell)
    ip.events.register('post_run_cell', cell.post_run_cell)
    ip.register_magic_function(s, 'line_cell')

def unload_ipython_extension(ip):
    ip.events.unregister('pre_run_cell', cell.pre_run_cell)
    ip.events.unregister('post_run_cell', cell.post_run_cell)
    del ip.magics_manager.magics['cell']['s']
    del ip.magics_manager.magics['line']['s']
