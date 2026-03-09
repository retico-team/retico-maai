import multiprocessing as mp
import queue
import warnings
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty
from threading import Event, Thread
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


class MaaiIU(GenericDictIU, ABC):
    """
    Abstract dictionary IU for holding MaAI output.
    """

    @abstractmethod
    def set_payload(self, provider_iter, payload): pass


class VAPIU(MaaiIU):
    """
    Dictionary IU holding voice activity projections (VAP).
    Payloads are treated as dictionaries of the form:

    {'p_now': [float, float], 'p_future': [float, float], 'vad': [float, float]}
    """

    @staticmethod
    def type():
        return "VAPIU"

    def set_payload(self, provider_iter, payload):
        self.payload = {
            f'{provider} ({idx + 1})': {
                'module': provider,
                'p_now': payload['p_now'][idx],
                'p_future': payload['p_future'][idx],
                'vad': payload['vad'][idx]
            } for idx, provider in provider_iter
        }


class BackchannelIU(MaaiIU):
    """
    Dictionary IU holding system backchannel timings.
    """

    @staticmethod
    def type():
        return "BackchannelIU"

    def set_payload(self, provider_iter, payload):
        self.payload = {
            'p_bc': payload['p_bc'],
        }


class NodIU(MaaiIU):
    """
    Dictionary IU holding system nod predictions.
    """

    @staticmethod
    def type():
        return "NodIU"

    def set_payload(self, provider_iter, payload):
        self.payload = {
            'p_bc': payload['p_bc'],
            'p_nod_short': payload['p_nod_short'],
            'p_nod_long': payload['p_nod_long'],
            'p_nod_long_p': payload['p_nod_long_p'],
        }


class MaaiBaseModule(AbstractModule):
    _MODE_DICT = {
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
        'bc': {
            'default': {
                'lang': 'en',
                'rate': 10,
            },
            'langs': [
                'jp',
                'en',
                'ch',
                'tri',
            ],
            'rates': [5, 10, 20],
        },
        'nod': {
            'default': {
                'lang': 'jp',
                'rate': 10,
            },
            'langs': [
                'jp',
            ],
            'rates': [5, 10, 20],
        }
    }
    _VIZ_TYPES = {
        'gui_bar': MaaiOutput.GuiBar,
        'gui_plot': MaaiOutput.GuiPlot,
        'console_bar': MaaiOutput.ConsoleBar,
    }
    _MAX_SAMPLES = 10

    def __init__(
            self,
            mode,
            lang=None,
            frame_rate=None,
            viz_type=None,
            max_samples=_MAX_SAMPLES,
            **kwargs
    ):
        """
        A module for MaAI turn-taking models, that takes in AudioIUs and
        outputs VAPIUs.

        :param mode: The mode to use for turn-taking predictions.
        :param lang: The language specification for the given mode. Defaults to the mode's default lang.
        :param frame_rate: The frame rate to use for the given mode. Defaults to the mode's default rate.
        :param viz_type: The MaaiOutput class to use.
        :param max_samples: The max number of audio chunks to queue for predictions.
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples

        if mode not in self._MODE_DICT.keys():
            raise RuntimeError(f"mode='{mode}' not one of {list(self._MODE_DICT.keys())}")

        if lang is None:
            lang = self._MODE_DICT[mode]['default']['lang']
        elif lang not in self._MODE_DICT[mode]['langs']:
            warnings.warn(f"lang='{lang}' not supported for {mode}")
            lang = self._MODE_DICT[mode]['default']['lang']
            print(f"WARNING -- Defaulting to lang='{lang}'")

        if frame_rate is None:
            frame_rate = self._MODE_DICT[mode]['default']['rate']
        elif frame_rate not in self._MODE_DICT[mode]['rates']:
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

        self._stop_event = Event()
        self._loop = Thread(target=self._maai_loop, daemon=True)

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
                if 'zmq_creator' in iu.meta_data:
                    self._channels[iu.meta_data['zmq_creator']][0].appendleft(iu)
                else:
                    self._channels[iu.creator][0].appendleft(iu)

    def _maai_loop(self):
        """Consumes IUs from each audio channel buffer and performs turn-taking predictions."""
        if self._viz_type:
            self._viz_process.start()
        while not self._stop_event.is_set():
            try:
                ch1, ch2 = list(self._channels.items())
                buf1, mic1 = ch1[1]
                buf2, mic2 = ch2[1]

                if ch2[0] != 'null':  # we're working with 2 channels
                    if buf1 and buf2:
                        mic1.put_chunk(np.frombuffer(buf1.pop().raw_audio, dtype=np.int16).astype(np.float32) / 32768.0)
                        mic2.put_chunk(np.frombuffer(buf2.pop().raw_audio, dtype=np.int16).astype(np.float32) / 32768.0)
                    else:
                        continue
                else:  # we're only working with one channel
                    if len(buf1) == 0:
                        sleep(_SLEEP_TIME)
                        continue

                    iu1 = buf1.pop()
                    if self.chunk_size is None:
                        self.zero_chunk = np.zeros(iu1.nframes, dtype=np.float32)
                    mic1.put_chunk(np.frombuffer(iu1.raw_audio, dtype=np.int16).astype(np.float32) / 32768.0)
                    mic2.put_chunk(self.zero_chunk)

                result = self.maai.result_dict_queue.get_nowait()
                if result:
                    new_iu = self.create_iu()
                    new_iu.set_payload(enumerate(self._channels.keys()), result)
                    um = UpdateMessage.from_iu(new_iu, UpdateType.ADD)
                    self.append(um)
                    if self._viz_type:
                        self._plot_queue.put(result)
            except Empty:
                sleep(_SLEEP_TIME)

    def shutdown(self):
        if self._viz_type:
            self._viz_liveness_w.close()
            self._plot_queue.put(None)
        self._stop_event.set()


class TurnTakingModule(MaaiBaseModule):

    def __init__(self, mode='vap_mc', **kwargs):
        if mode not in ['vap_mc', 'vap']: raise ValueError(f"Mode must be 'vap' or 'vap_mc', got '{mode}'")
        super().__init__(mode=mode, **kwargs)

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


class BackchannelModule(MaaiBaseModule):

    def __init__(self, **kwargs):
        super().__init__(mode='bc', **kwargs)

    @staticmethod
    def name():
        return "BackchannelModule"

    @staticmethod
    def description():
        return "Backchannel module for MaAI models"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return BackchannelIU


class NodPredictionModule(MaaiBaseModule):

    def __init__(self, **kwargs):
        super().__init__(mode='nod', **kwargs)

    @staticmethod
    def name():
        return "NodPredictionModule"

    @staticmethod
    def description():
        return "Nod prediction module for MaAI models"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return NodIU
