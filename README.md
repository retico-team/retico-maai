# retico-maai
    
Retico bindings for [MaAI](https://github.com/MaAI-Kyoto/MaAI),
providing turn-taking, backchannel, and nodding inference.

### Installation and requirements ###

The package can be installed using the following:  
`pip install git+https://github.com/retico-team/retico-maai`

### Turn-Taking Example ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_maai import TurnTakingModule

mic = MicrophoneModule()
maai = TurnTakingModule(mode='vap_mc', lang='en', frame_rate=10)
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
Using this module with [retico-zmq](https://github.com/retico-team/retico-zmq) can cause time alignment issues. To avoid
this, the runner housing the TurnTakingModule should be given ample time to perform setup before any other runners it
depends on for input are started. Additionally, the clocks for each machine hosting a runner should be synchronized as
this module uses timestamps for synchronizing audio input.

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
