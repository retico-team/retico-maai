import warnings
from queue import Queue, Empty
from threading import Lock, Thread
from time import sleep

import numpy as np
from maai import Maai, MaaiInput
from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.audio import AudioIU
from retico_core.dialogue import GenericDictIU
from torch.cuda import is_available


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
    _MAX_SAMPLES = 10
    _SLEEP_TIME = 0.05

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
            max_samples=_MAX_SAMPLES,
            **kwargs
    ):
        """
        A module for MaAI turn-taking models, that takes in AudioIUs and
        outputs VAPIUs.

        :param mode: The mode to use for turn-taking predictions.
        :param lang: The language specification for the given mode.
        :param frame_rate: The frame rate to use for the given mode.
        :param max_samples: The max number of audio chunks to queue for predictions.
        """
        super().__init__(**kwargs)
        if max_samples != self._MAX_SAMPLES: self._MAX_SAMPLES = max_samples
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
        self.mics = {}
        self.maai = None
        self.chunk_size = None
        self.zero_chunk = None
        self._lock = Lock()
        self._loop = Thread(target=self._maai_loop, daemon=True)
        self._loop_active = True
        self._channel_buffers = {}

    def setup(self):
        for buf in self._left_buffers:
            if buf.provider not in self._channel_buffers.keys():
                self._channel_buffers[buf.provider] = Queue(self._MAX_SAMPLES)
                self.mics[buf.provider] = MaaiInput.Chunk()

        if len(self._channel_buffers.keys()) == 0:
            raise RuntimeWarning("No inputs detected")
        elif len(self._channel_buffers.keys()) == 1:
            self._channel_buffers['zero'] = Queue(self._MAX_SAMPLES)
            self.mics['zero'] = MaaiInput.Chunk()
        elif len(self._channel_buffers.keys()) > 2:
            raise RuntimeError(f"Expected 1 or 2 channels, found {len(self._channel_buffers.keys())}")

        self.maai = Maai(
            mode=self.mode,
            lang=self.lang,
            frame_rate=self.frame_rate,
            audio_ch1=list(self.mics.values())[0],
            audio_ch2=list(self.mics.values())[1],
            device=self.device
        )
        self.maai.start()
        self._loop.start()

    def process_update(self, um):
        for iu, ut in um:
            if ut == UpdateType.ADD:
                self._channel_buffers[iu.creator].put(iu)

    def _maai_loop(self):
        while True:
            try:
                iu = None
                for provider, buf in self._channel_buffers.items():
                    if self.chunk_size is None and provider == 'zero':
                        # zeroed mic will always come second if it exists, so iu will already be assigned
                        self.chunk_size = round(iu.nframes * iu.sample_width)
                        self.zero_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                        self.mics[provider].put_chunk(self.zero_chunk)
                    elif provider == 'zero':
                        self.mics[provider].put_chunk(self.zero_chunk)
                    else:
                        iu = buf.get()
                        self.mics[provider].put_chunk(iu.raw_audio)

                result = self.maai.result_dict_queue.get_nowait()
                if result:
                    new_iu = self.create_iu()
                    new_iu.payload = {
                        'p_now': result['p_now'],
                        'p_future': result['p_future'],
                        'vad': result['vad'],
                    }
                    um = UpdateMessage.from_iu(new_iu, UpdateType.ADD)
                    self.append(um)
            except Empty:
                sleep(self._SLEEP_TIME)
            finally:
                self._lock.acquire()
                if not self._loop_active: break
                self._lock.release()

    def shutdown(self):
        self._lock.acquire()
        self._loop_active = False
        self._lock.release()
        self._loop.join()
