# retico-maai
    
Retico bindings for [MaAI](https://github.com/MaAI-Kyoto/MaAI),
providing turn-taking, backchannel, and nodding inference.

## Installation and Requirements

To use the MaAI modules you first need to install the retico-maai package:
* Install retico-maai:
```pip install retico-maai```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-maai.git```

## Modules

All three modules subscribe to `AudioIU`. Each audio module subscribed to them is used as a separate speaker channel, up to two; with a single microphone, the second channel is filled with silence.

### `TurnTakingModule`
Predicts turn-taking with Voice Activity Projection (VAP) and outputs a `VAPIU`, whose payload holds, for each channel, the current and future speaking probabilities (`p_now`, `p_future`) and voice activity (`vad`).

#### Arguments:
* `mode` (str): `"vap_mc"` or `"vap"`; defaults to `"vap_mc"`
* `lang` (str): the model language: `"en"`, `"jp"`, `"ch"` or `"tri"` (trilingual), or their `_kyoto` variants; defaults to `"en"`
* `frame_rate` (int): predictions per second: `5`, `10` or `20`; defaults to `10`
* `viz_type` (str): optional live visualization: `"gui_plot"`, `"gui_bar"` or `"console_bar"`; defaults to `None` (no visualization)

### `BackchannelModule`
Predicts when the system should produce a backchannel (e.g. "yeah", "uh-huh") and outputs a `BackchannelIU` with the probability `p_bc`.

#### Arguments:
* `lang` (str): `"en"`, `"jp"`, `"ch"` or `"tri"`; defaults to `"en"`
* `frame_rate` (int): `5`, `10` or `20`; defaults to `10`
* `viz_type` (str): same as for `TurnTakingModule`; defaults to `None`

### `NodPredictionModule`
Predicts when the system should nod and outputs a `NodIU` with the probabilities `p_bc`, `p_nod_short`, `p_nod_long` and `p_nod_long_p`. Only Japanese (`"jp"`) is available.

#### Arguments:
* `frame_rate` (int): `5`, `10` or `20`; defaults to `10`
* `viz_type` (str): same as for `TurnTakingModule`; defaults to `None`

### Turn-Taking Example ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_maai import TurnTakingModule

mic = MicrophoneModule()
maai = TurnTakingModule(mode='vap_mc', lang='en', frame_rate=10, viz_type='gui_plot')
debug = DebugModule(print_payload_only=True)

mic.subscribe(maai)
maai.subscribe(debug)

mic.run()
maai.run()
debug.run()

input()

mic.stop()
maai.stop()
debug.stop()
```

#### *Note ####
*Using this module with microphone input via [retico-zmq](https://github.com/retico-team/retico-zmq) can cause time 
alignment issues, and is experimental. To avoid this, the runner housing the TurnTakingModule and the runner housing 
the foreign microphone module need to be manually synchronized by coordinating when each is started.*

### Backchannel Example ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_maai import BackchannelModule

mic = MicrophoneModule()
maai = BackchannelModule(lang='en', frame_rate=10, viz_type='gui_plot')
debug = DebugModule(print_payload_only=True)

mic.subscribe(maai)
maai.subscribe(debug)

mic.run()
maai.run()
debug.run()

input()

mic.stop()
maai.stop()
debug.stop()
```

#### *Note ####
*The backchannel module is intended to be used with a single microphone input.*

### Nod Prediction Example ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_maai import NodPredictionModule

mic = MicrophoneModule()
maai = NodPredictionModule(viz_type='gui_plot')
debug = DebugModule(print_payload_only=True)

mic.subscribe(maai)
maai.subscribe(debug)

mic.run()
maai.run()
debug.run()

input()

mic.stop()
maai.stop()
debug.stop()
```

#### *Note ####
*The nod prediction module is intended to be used with a single microphone input.*

### Citations ###
Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze
**Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection**
International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2024
https://arxiv.org/abs/2401.04868
```
@inproceedings{inoue2024iwsds,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
    year = {2024},
    url = {https://arxiv.org/abs/2401.04868},
}
```
Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze
**Multilingual Turn-taking Prediction Using Voice Activity Projection**
Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pages 11873-11883, 2024
https://aclanthology.org/2024.lrec-main.1036/
```
@inproceedings{inoue2024lreccoling,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Multilingual Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {Proceedings of the Joint International Conference on Computational Linguistics and Language Resources and Evaluation (LREC-COLING)},
    pages = {11873--11883},
    year = {2024},
    url = {https://aclanthology.org/2024.lrec-main.1036/},
}
```
Koji Inoue, Yuki Okafuji, Jun Baba, Yoshiki Ohira, Katsuya Hyodo, Tatsuya Kawahara
**A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment**
https://www.arxiv.org/abs/2503.06241
```
@misc{inoue2025noisevap,
    author = {Koji Inoue and Yuki Okafuji and Jun Baba and Yoshiki Ohira and Katsuya Hyodo and Tatsuya Kawahara},
    title = {A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment},
    year = {2025},
    note = {arXiv:2503.06241},
    url = {https://www.arxiv.org/abs/2503.06241},
}
```
Koji Inoue, Divesh Lala, Gabriel Skantze, Tatsuya Kawaharaa
**Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection**
https://aclanthology.org/2025.naacl-long.367/
```
@inproceedings{inoue2025vapbc,
    author = {Koji Inoue and Divesh Lala and Gabriel Skantze and Tatsuya Kawahara},
    title = {Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection},
    booktitle = {Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
    pages = {7171--7181},
    year = {2025},
    url = {https://aclanthology.org/2025.naacl-long.367/},
}
```
