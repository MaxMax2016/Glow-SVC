<div align="center">
<h1> Max's Singing Voice Conversion, Just for Playing! </h1>
As this name show, this is a personal project. [WIP]
</div>

## Setup Environment
1. Install project dependencies

    ```shell
    pip install -r requirements.txt
    ```

2. Download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`.

3. Download [hubert_soft model](https://github.com/bshall/hubert/releases/tag/v0.1)，put `hubert-soft-0d54a1f4.pt` into `hubert_pretrain/`.

4. Download pretrain model, and put it into `vits_pretrain/`.
    ```shell
    python svc_inference.py --config configs/base.yaml --model ./vits_pretrain/svc.pretrain.pth --spk ./configs/singers/singer0001.npy --wave test.wav
    ```

## Dataset preparation
Put the dataset into the `data_raw` directory following the structure below.
```
data_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## Data preprocessing
After preprocessing you will get an output with following structure.
```
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── mel
│    └── speaker0
│    │      ├── 000001.mel.pt
│    │      └── 000xxx.mel.pt
│    └── speaker1
│           ├── 000001.mel.pt
│           └── 000xxx.mel.pt
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── hubert
│    └── speaker0
│    │      ├── 000001.vec.npy
│    │      └── 000xxx.vec.npy
│    └── speaker1
│           ├── 000001.vec.npy
│           └── 000xxx.vec.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

1.  Re-sampling
    - Generate audio with a sampling rate of 16000Hz in `./data_svc/waves-16k` 
    ```
    python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-16k -s 16000
    ```
    
    - Generate audio with a sampling rate of 32000Hz in `./data_svc/waves-32k`
    ```
    python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-32k -s 32000
    ```
2. Use 16K audio to extract pitch
    ```
    python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch
    ```
3. use 32k audio to extract mel
    ```
    python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/mel
    ``` 
4. Use 16K audio to extract hubert
    ```
    python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert
    ```
5. Use 16k audio to extract timbre code
    ```
    python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker
    ```
6. Extract the average value of the timbre code for inference
    ```
    python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer
    ``` 
8. Use 32k audio to generate training index
    ```
    python prepare/preprocess_train.py
    ```
9. Training file debugging
    ```
    python prepare/preprocess_zzz.py
    ```

## Train
1. Start training
   ```
   python svc_trainer.py -c configs/base.yaml -n svc
   ``` 
2. Resume training
   ```
   python svc_trainer.py -c configs/base.yaml -n svc -p chkpt/svc/***.pth
   ```
3. Log visualization
   ```
   tensorboard --logdir logs/
   ```

## Loss
mel_loss should be less than 0.45

## Inference

1. Export inference model
   ```
   python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/svc/***.pt
   ```

2. Inference
    - if there is no need to adjust `f0`, just run the following command.
        ```
        python svc_inference.py --config configs/base.yaml --model svc.pth --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav --shift 0
        ```
    - if `f0` will be adjusted manually, follow the steps:

        1. use hubert to extract content vector
            ```
            python hubert/inference.py -w test.wav -v test.vec.npy
            ```
        2. extract the F0 parameter to the csv text format
            ```
            python pitch/inference.py -w test.wav -p test.csv
            ```
        3. final inference
            ```
            python svc_inference.py --config configs/base.yaml --model svc.pth --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav --vec test.vec.npy --pit test.csv --shift 0
            ```

3. Convert mel to wave
    ```
    python svc_inference_wave.py --mel svc_out.mel.pt --pit svc_tmp.pit.csv
    ```

4. Debug mel for wave
   
    ```
    python spec/inference.py -w test.wav -m test.mel.pt
    ```

## Code sources and references

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [paper](https://arxiv.org/abs/2106.07889)

https://github.com/mozilla/TTS

https://github.com/bshall/soft-vc

https://github.com/maxrmorrison/torchcrepe

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://github.com/hcy71o/SNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://github.com/ubisoft/ubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://arxiv.org/abs/2305.05401)

[Adversarial Speaker Disentanglement Using Unannotated External Data for Self-supervised Representation Based Voice Conversion](https://arxiv.org/pdf/2305.09167.pdf)

[Speaker normalization (GRL) for self-supervised speech emotion recognition](https://arxiv.org/abs/2202.01252)
