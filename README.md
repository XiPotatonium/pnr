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

#### Datasets

+ ACE04: https://catalog.ldc.upenn.edu/LDC2005T09
+ ACE05: https://catalog.ldc.upenn.edu/LDC2006T06
+ KBP17: https://catalog.ldc.upenn.edu/LDC2017D55
+ GENIA: http://www.geniaproject.org/genia-corpus
+ CoNLL03: https://data.deepai.org/conll2003.zip


Data format:
```json
{
    "tokens": ["Others", ",", "though", ",", "are", "novices", "."],
    "entities": [{"type": "PER", "start": 0, "end": 1}, {"type": "PER", "start": 5, "end": 6}], "relations": [], "org_id": "CNN_IP_20030328.1600.07",
    "ltokens": ["WOODRUFF", "We", "know", "that", "some", "of", "the", "American", "troops", "now", "fighting", "in", "Iraq", "are", "longtime", "veterans", "of", "warfare", ",", "probably", "not", "most", ",", "but", "some", ".", "Their", "military", "service", "goes", "back", "to", "the", "Vietnam", "era", "."],
    "rtokens": ["So", "what", "is", "it", "like", "for", "them", "to", "face", "combat", "far", "from", "home", "?", "For", "an", "idea", ",", "here", "is", "CNN", "'s", "Candy", "Crowley", "with", "some", "war", "stories", "."]
}
```

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

## Appendix

Please refer to [Appendix](doc/appendix.md)

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2204.12732,
  doi = {10.48550/ARXIV.2204.12732},

  url = {https://arxiv.org/abs/2204.12732},

  author = {Wu, Shuhui and Shen, Yongliang and Tan, Zeqi and Lu, Weiming},

  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Propose-and-Refine: A Two-Stage Set Prediction Network for Nested Named Entity Recognition},

  publisher = {arXiv},

  year = {2022},

  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
