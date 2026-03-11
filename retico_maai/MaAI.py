import multiprocessing as mp
import queue
import warnings
from abc import ABC, abstractmethod
from queue import Empty
from threading import Event, Lock, Thread
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
            f'{mod_id} ({idx + 1})': {
                'module': mod_id,
                'p_now': payload['p_now'][idx],
                'p_future': payload['p_future'][idx],
                'vad': payload['vad'][idx]
            } for idx, mod_id in provider_iter
        }


class BackchannelIU(MaaiIU):
    """
    Dictionary IU holding system backchannel timings.
    Payloads are treated as dictionaries of the form:

    {'p_bc': float}
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
    Payloads are treated as dictionaries of the form:

    {'p_bc': float, 'p_nod_short': float, 'p_nod_long': float, 'p_nod_long_p': float}
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

    def __init__(
            self,
            mode,
            lang=None,
            frame_rate=None,
            viz_type=None,
            **kwargs
    ):
        """
        Abstract base module for MaAI models.

        :param mode: The prediction mode to use (i.e. vap | bc | nod).
        :param lang: The language specification for the given mode. Defaults are set in _MODE_DICT per mode.
        :param frame_rate: The frame rate to use for the given mode. Defaults are set in _MODE_DICT.
        :param viz_type: The MaaiOutput class to use (i.e. plot | bar | console).
        """
        super().__init__(**kwargs)

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

        self.zero_chunk = None
        self._channels = {}
        self._channel_lock = Lock()
        self.channel_cnt = 0
        self.ch1, self.ch2 = MaaiInput.Chunk(), MaaiInput.Chunk()
        self.maai = Maai(
            mode=self.mode,
            lang=self.lang,
            frame_rate=self.frame_rate,
            audio_ch1=self.ch1,
            audio_ch2=self.ch2,
            device=self.device
        )
        self.maai.start()

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

        self._stop_event = Event()
        self._loop = Thread(target=self._maai_loop, daemon=True)
        self._loop.start()

    def process_update(self, um):
        for iu, ut in um:
            if ut == UpdateType.ADD:
                with self._channel_lock:
                    try:
                        chunk = np.frombuffer(iu.raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        self._channels[iu.creator_id].put_chunk(chunk)
                        if self.channel_cnt == 1:
                            self.ch2.put_chunk(self.zero_chunk)
                    except KeyError as e:
                        match self.channel_cnt:
                            case 0:
                                self._channels[iu.creator_id] = self.ch1
                                self.channel_cnt += 1
                                self.zero_chunk = np.zeros(iu.nframes, dtype=np.float32)
                                print(f"Binding channel 1 to {iu.creator_id}")
                            case 1:
                                self._channels[iu.creator_id] = self.ch2
                                self.channel_cnt += 1
                                print(f"Binding channel 2 to {iu.creator_id}")
                            case _:
                                raise RuntimeError(f"Too many channels detected -- {e}")

    def _maai_loop(self):
        if self._viz_type:
            self._viz_process.start()
        while not self._stop_event.is_set():
            try:
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
        self._loop.join()
        self.maai.stop()
        if self.maai._worker_thread is not None:
            self.maai._worker_thread.join()


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
