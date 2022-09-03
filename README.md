# Propose-and-Refine: A Two-Stage Set Prediction Network for Nested Named Entity Recognition

## Overview

![Overview](doc/model.png "Overview of PnRNet. In the propose stage, PnRNet computes span representations and generates coarse entity proposals with a span-based predictor. In the refine stage, the proposals are refined through a transformer decoder and finally are used to re-predict boundaries and entity classes. We collect multi-scale features from span features generated in the propose stage to provide hierarchical contextual information in proposal refinement. For simplicity of demonstration, we show a PnRNet with span enumeration length limited to $L=4$.")

## Usage

### Environment setup and preparation

```bash
conda create --name pnr --file requirements.txt
```

* Place dataset under folder `data`
* Download pre-trained word embedding and set `wordvec_path` in the configuration file to the path of word embedding file.
* Download pre-trained language model (from huggingface) and set `model_path` and `tokenizer_path` in the configuration file to the path of the pre-trained language model.

### Train

We train our model using RTX-3090 when training with ACE-04, ACE-05, KBP-17.
And using RTX-A6000 when training with GENIA and Conll-03.

```bash
python main.py train --config cfg/ace05/train.conf
```

### Evaluate

Set `model_path` and `tokenzier_path` to the checkpoint dir before running.

We provide our checkpoints on [ACE04 and ACE05](https://drive.google.com/drive/folders/1Hg3OoRDzOiCpUPPGI85H3OVzuIIRBBa3?usp=sharing)

```bash
python main.py eval --config cfg/ace05/eval.conf
```

## Datasets

+ ACE04: https://catalog.ldc.upenn.edu/LDC2005T09
+ ACE05: https://catalog.ldc.upenn.edu/LDC2006T06
+ KBP17: https://catalog.ldc.upenn.edu/LDC2017D55
+ GENIA: http://www.geniaproject.org/genia-corpus
+ CoNLL03: https://data.deepai.org/conll2003.zip


### Format:

train/dev/test dataset is a `.json` file consists of a list of samples.
Each sample is a dictionary with the following format:

```json
{
    "tokens": ["Others", ",", "though", ",", "are", "novices", "."],
    "entities": [{"type": "PER", "start": 0, "end": 1}, {"type": "PER", "start": 5, "end": 6}], "relations": [], "org_id": "CNN_IP_20030328.1600.07",
    "ltokens": ["WOODRUFF", "We", "know", "that", "some", "of", "the", "American", "troops", "now", "fighting", "in", "Iraq", "are", "longtime", "veterans", "of", "warfare", ",", "probably", "not", "most", ",", "but", "some", ".", "Their", "military", "service", "goes", "back", "to", "the", "Vietnam", "era", "."],
    "rtokens": ["So", "what", "is", "it", "like", "for", "them", "to", "face", "combat", "far", "from", "home", "?", "For", "an", "idea", ",", "here", "is", "CNN", "'s", "Candy", "Crowley", "with", "some", "war", "stories", "."]
}
```

Entity types should be specified in a `type.json` file:

```json
{
  "entities": {"GPE": {"short": "GPE", "verbose": "GPE"}, "ORG": {"short": "ORG", "verbose": "ORG"}, "PER": {"short": "PER", "verbose": "PER"}, "LOC": {"short": "LOC", "verbose": "LOC"}, "FAC": {"short": "FAC", "verbose": "FAC"}, "VEH": {"short": "VEH", "verbose": "VEH"}, "WEA": {"short": "WEA", "verbose": "WEA"}}
}
```

If you want to use part-of-speech tags as additional features,
you should add a `pos` field for each sample like:

```json
"pos": ["PROPN", "PROPN", "PROPN", ",", "PROPN", ",", "PROPN", "NOUN"]
```

And you should specify all pos tags in a `pos.json` file (`pos.json` and `type.json` should be placed in the same folder), the value of each tag represents the tag frequency:

```json
{"PROPN": 15000, ",": 8169, "NOUN": 34923, "DET": 16595, "ADJ": 11268, "CCONJ": 4470, "VERB": 19302, "ADV": 6187, "AUX": 7046, "ADP": 17412, "NUM": 4154, ".": 8245, "X": 183, "(": 284, ")": 296, "PART": 5451, "SCONJ": 2225, "\"": 654, "PRON": 7613, "'": 1974, ":": 231, ";": 73, "-": 1366, "SYM": 437, "`": 1401, "INTJ": 194, "?": 203, "!": 26, "ReplayTV": 1, "/": 3, "_": 2, "s": 2, "]": 2, "}": 1, "{": 1}
```

## Appendix

Please refer to [Appendix](doc/appendix.md)

## Citation

```
@inproceedings{ijcai2022p613,
  title     = {Propose-and-Refine: A Two-Stage Set Prediction Network for Nested Named Entity Recognition},
  author    = {Wu, Shuhui and Shen, Yongliang and Tan, Zeqi and Lu, Weiming},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4418--4424},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/613},
  url       = {https://doi.org/10.24963/ijcai.2022/613},
}
```
