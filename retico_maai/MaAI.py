import multiprocessing as mp
import queue
import warnings
from collections import deque
from queue import Empty
from threading import Lock, Thread
from time import sleep

import numpy as np
from maai import Maai, MaaiInput, MaaiOutput
from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.audio import AudioIU
from retico_core.dialogue import GenericDictIU
from torch.cuda import is_available

_SLEEP_TIME = 0.05


def _run_visualization(viz_type, plot_queue, liveness):
    output = viz_type() if viz_type is not None else None
    while not liveness.poll():
        try:
            result = plot_queue.get(timeout=0.05)
            if result is None: break
            if output is not None: output.update(result)
        except queue.Empty:
            sleep(_SLEEP_TIME)
            continue


class VAPIU(GenericDictIU):
    """
    Dictionary IU holding voice activity projections (VAP).
    Payloads are treated as dictionaries of the form:

    {'p_now': [float, float], 'p_future': [float, float], 'vad': [float, float]}
    """

    @staticmethod
    def type():
        return "VAPIU"

    def __init__(
            self,
            creator=None,
            iuid=0,
            previous_iu=None,
            grounded_in=None,
            payload=None,
            **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=payload,
            **kwargs
        )


class TurnTakingModule(AbstractModule):
    _MODE_DICT = {
        'default': 'vap_mc',
        'vap_mc': {
            'default': {
                'lang': 'en',
                'rate': 10,
            },
            'langs': [
                'jp',
                'jp_kyoto',
                'en',
                'en_kyoto',
                'ch',
                'ch_kyoto',
                'tri',
                'tri_kyoto',
            ],
            'rates': [5, 10, 20],
        },
        'vap': {
            'default': {
                'lang': 'en',
                'rate': 10,
            },
            'langs': [
                'jp',
                'jp_kyoto',
                'en',
                'en_kyoto',
                'ch',
                'ch_kyoto',
                'tri',
                'tri_kyoto',
            ],
            'rates': [5, 10, 20],
        },
    }
    _VIZ_TYPES = {
        'gui_bar': MaaiOutput.GuiBar,
        'gui_plot': MaaiOutput.GuiPlot,
        'console_bar': MaaiOutput.ConsoleBar,
    }
    _MAX_SAMPLES = 10
    _DESYNC = 0.1  # desync tolerance

    @staticmethod
    def name():
        return "TurnTakingModule"

    @staticmethod
    def description():
        return "Turn-taking module for MaAI turn-taking models"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return VAPIU

    def __init__(
            self,
            mode=f"{_MODE_DICT['default']}",
            lang=f"{_MODE_DICT[_MODE_DICT['default']]['default']['lang']}",
            frame_rate=f"{_MODE_DICT[_MODE_DICT['default']]['default']['rate']}",
            viz_type=None,
            max_samples=_MAX_SAMPLES,
            desync_tolerance=_DESYNC,
            **kwargs
    ):
        """
        A module for MaAI turn-taking models, that takes in AudioIUs and
        outputs VAPIUs.

        :param mode: The mode to use for turn-taking predictions.
        :param lang: The language specification for the given mode.
        :param frame_rate: The frame rate to use for the given mode.
        :param viz_type: The MaaiOutput class to use.
        :param max_samples: The max number of audio chunks to queue for predictions.
        :param desync_tolerance: The tolerance for audio desynchronization.
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples
        self.desync_tolerance = desync_tolerance
        if mode not in self._MODE_DICT.keys():
            warnings.warn(f"mode='{mode}' not supported")
            mode = self._MODE_DICT['default']
            lang = self._MODE_DICT[mode]['default']['lang']
            frame_rate = self._MODE_DICT[mode]['default']['rate']
            print(f"WARNING -- Defaulting to mode='{mode}'")
            print(f"WARNING -- Defaulting to lang='{lang}'")
            print(f"WARNING -- Defaulting to frame_rate={frame_rate}")
        if lang not in self._MODE_DICT[mode]['langs']:
            warnings.warn(f"lang='{lang}' not supported for {mode}")
            lang = self._MODE_DICT[mode]['default']['lang']
            print(f"WARNING -- Defaulting to lang='{lang}'")
        if frame_rate not in self._MODE_DICT[mode]['rates']:
            warnings.warn(f"frame_rate={frame_rate} not supported for {mode}")
            frame_rate = self._MODE_DICT[mode]['default']['rate']
            print(f"WARNING -- Defaulting to frame_rate={frame_rate}")

        self.mode = mode
        self.lang = lang
        self.frame_rate = frame_rate
        self.device = 'cuda' if is_available() else 'cpu'

        self.maai = None
        self.chunk_size = None
        self.zero_chunk = None
        self._channels = {}

        self._lock = Lock()
        self._loop = Thread(target=self._maai_loop, daemon=True)
        self._loop_active = True

        self._plot_queue = mp.Queue()
        self._viz_type = self._VIZ_TYPES[viz_type] if viz_type in self._VIZ_TYPES else None
        _viz_liveness_r, self._viz_liveness_w = mp.Pipe(duplex=False)
        self._viz_process = mp.Process(
            target=_run_visualization,
            args=(self._viz_type, self._plot_queue, _viz_liveness_r),
            daemon=True,
            name="MaAI-Visualization"
        )

    def setup(self):
        """
        Infers audio sources from subscribed modules and starts MaAI inference loop.
        Initializes audio channel buffers and MaAI microphones for storing IUs and
        performing inference.
        """
        for buf in self._left_buffers:
            if buf.provider not in self._channels.keys():
                self._channels[buf.provider] = [deque(maxlen=self.max_samples), MaaiInput.Chunk()]

        if len(self._channels.keys()) == 0:
            raise RuntimeError("No inputs detected")
        elif len(self._channels.keys()) == 1:
            self._channels['null'] = [deque(maxlen=self.max_samples), MaaiInput.Chunk()]
        elif len(self._channels.keys()) > 2:
            raise RuntimeError(f"Expected 1 or 2 channels, found {len(self._channels.keys())}")

        ch1, ch2 = self._channels.keys()
        ch1, ch2 = self._channels[ch1][1], self._channels[ch2][1]
        self.maai = Maai(
            mode=self.mode,
            lang=self.lang,
            frame_rate=self.frame_rate,
            audio_ch1=ch1,
            audio_ch2=ch2,
            device=self.device
        )
        self.maai.start()
        self._loop.start()

    def process_update(self, um):
        for iu, ut in um:
            if ut == UpdateType.ADD:
                with self._lock: self._channels[iu.creator][0].appendleft(iu)

    def _maai_loop(self):
        """Consumes IUs from each audio channel buffer and performs turn-taking predictions."""
        self._viz_process.start()
        while True:
            try:
                chan1_vals, chan2_vals = list(self._channels.items())
                provider1, provider2 = chan1_vals[0], chan2_vals[0]
                lst1, lst2 = chan1_vals[1], chan2_vals[1]
                buf1, mic1 = lst1
                buf2, mic2 = lst2

                if provider2 != 'null':  # we're working with 2 channels
                    if len(buf1) == 0 or len(buf2) == 0:
                        sleep(_SLEEP_TIME)
                        continue

                    with self._lock:
                        iu1, iu2 = buf1.pop(), buf2.pop()

                        # calculate chunk_size and zero_chunk if not already
                        if self.chunk_size is None:
                            self.chunk_size = round(iu1.nframes * iu1.sample_width)
                            self.zero_chunk = np.zeros(self.chunk_size, dtype=np.float32)

                        ts1 = iu1.meta_data['producing_module_timestamp']
                        ts2 = iu2.meta_data['producing_module_timestamp']
                        diff = ts1 - ts2

                        if abs(diff) > self.desync_tolerance and diff < 0:
                            # ts1 is older than ts2, so process ts1 and put ts2 back
                            buf2.append(iu2)
                            mic1.put_chunk(iu1.raw_audio)
                            mic2.put_chunk(self.zero_chunk)
                        elif abs(diff) > self.desync_tolerance and diff > 0:
                            # ts1 is younger than ts2, so put ts1 back and process ts2
                            buf1.append(iu1)
                            mic1.put_chunk(self.zero_chunk)
                            mic2.put_chunk(iu2.raw_audio)
                        else:
                            # ts1 and ts2 are in sync, so process both
                            mic1.put_chunk(iu1.raw_audio)
                            mic2.put_chunk(iu2.raw_audio)
                else:  # we're only working with one channel
                    if len(buf1) == 0:
                        sleep(_SLEEP_TIME)
                        continue

                    iu1 = buf1.pop()

                    # calculate chunk_size and zero_chunk if not already
                    if self.chunk_size is None:
                        self.chunk_size = round(iu1.nframes * iu1.sample_width)
                        self.zero_chunk = np.zeros(self.chunk_size, dtype=np.float32)

                    mic1.put_chunk(iu1.raw_audio)
                    mic2.put_chunk(self.zero_chunk)

                result = self.maai.result_dict_queue.get_nowait()
                if result:
                    new_iu = self.create_iu()
                    new_iu.payload = {
                        f'{provider} ({idx + 1})': {
                            'module': provider,
                            'p_now': result['p_now'][idx],
                            'p_future': result['p_future'][idx],
                            'vad': result['vad'][idx]
                        } for idx, provider in enumerate(self._channels.keys())
                    }
                    um = UpdateMessage.from_iu(new_iu, UpdateType.ADD)
                    self.append(um)
                    self._plot_queue.put(result)
            except Empty:
                sleep(_SLEEP_TIME)
            finally:
                with self._lock:
                    if not self._loop_active: break

    def shutdown(self):
        self._viz_liveness_w.close()
        self._plot_queue.put(None)
        with self._lock:
            self._loop_active = False
