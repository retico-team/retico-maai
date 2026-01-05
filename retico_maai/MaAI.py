import numpy as np
import queue
from threading import Lock, Thread
from time import sleep
from maai import Maai, MaaiInput
from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.dialogue import GenericDictIU
from retico_core.audio import AudioIU
from torch.cuda import is_available


class VAPIU(GenericDictIU):
    """Dictionary IU holding voice activity projections (VAP)."""

    @staticmethod
    def type():
        return "VAP IU"

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


class MaAI(AbstractModule):
    _MODE_LIST = [
        'vap',
        'vap_mc',
    ]
    _LANG_LIST = [
        'jp',
        'jp_kyoto',
        'en',
        'en_kyoto',
        'ch',
        'ch_kyoto',
        'tri',
        'tri_kyoto',
    ]
    _FRAME_RATES = [
        5,
        10,
        20,
    ]

    @staticmethod
    def name():
        return "MaAI"

    @staticmethod
    def description():
        return "Interfacing module for the MaAI software"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return VAPIU

    def __init__(
            self,
            mode='vap_mc',
            lang='en',
            frame_rate=10,
            max_samples=10,
            **kwargs
    ):
        super().__init__(**kwargs)
        if mode not in self._MODE_LIST:
            f"mode='{mode}' not supported. Defaulting to '{mode:='vap_mc'}'."
        if lang not in self._LANG_LIST:
            f"lang='{lang}' not supported. Defaulting to '{lang:='en'}'."
        if frame_rate not in self._FRAME_RATES:
            f"frame_rate={frame_rate} not supported. Defaulting to {frame_rate:=10}."

        device = 'cuda' if is_available() else 'cpu'
        self.chunk_size = None
        self.zero_chunk = None
        self.mic = MaaiInput.Chunk()
        self.zero = MaaiInput.Chunk()
        self.maai = Maai(
            mode=mode,
            lang=lang,
            frame_rate=frame_rate,
            audio_ch1=self.mic,
            audio_ch2=self.zero,
            device=device
        )
        self.maai.start()
        self.queue = queue.Queue(max_samples)
        self._lock = Lock()
        self._loop = Thread(target=self._maai_loop, daemon=True)
        self._loop_active = True
        self._loop.start()

    def process_update(self, um):
        for iu, ut in um:
            if ut == UpdateType.ADD:
                self.queue.put(iu)

    def _maai_loop(self):
        while True:
            try:
                iu = self.queue.get_nowait()

                if self.chunk_size is None:
                    self.chunk_size = round(iu.nframes * iu.sample_width)
                    self.zero_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                if len(iu.raw_audio) != self.chunk_size: continue

                self.mic.put_chunk(iu.raw_audio)
                self.zero.put_chunk(self.zero_chunk)
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
            except queue.Empty:
                sleep(0.1)
                continue
            finally:
                self._lock.acquire()
                if not self._loop_active: break
                self._lock.release()

    def shutdown(self):
        self._lock.acquire()
        self._loop_active = False
        self._lock.release()
        self._loop.join()
