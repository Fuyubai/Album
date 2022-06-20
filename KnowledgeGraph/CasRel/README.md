## CasRel
That code is a simple reproduction of [CasRel](http://export.arxiv.org/pdf/1909.03227/n)

I don't refer native implementation which use tensorflow, rather this [  
CasRel-pytorch-reimplement](https://github.com/longlongman/CasRel-pytorch-reimplement)

Note: Just for learning, the inference performance may not stasify the production environment, though it has beed optimized 

### Quick Start
```python
python train.py
```

### Dataset
- [CMED](biendata.xyz/competition/chip_2020_2/): CHIP-2020 中文医学文本实体关系抽取

### Metrics
- use train set to train, and use dev set to eval

||My implementation|CasRel-pytorch-reimplement|
| :--: | :--:|:--:|
|10epochs|0.4431|0.45|
|20epochs|0.4816|0.48|

- use dev set to train, and use dev set to eval

||My implementation|CasRel-pytorch-reimplement|
| :--: | :--:|:--:|
|10epochs|0.4431|0.37|
|20epochs|0.7277|0.62|

### Conclude
- When I use same training process like CasRel-pytorch-reimplement, I found it hard to fit, maybe My implementation' s backbone is [Tiny_Bert_4L](https://huggingface.co/huawei-noah/TinyBERT_4L_zh), whereas CasRel-pytorch-reimplement uses [chinese_wwm_pytorch](https://github.com/ymcui/Chinese-BERT-wwm) as backbone, which is larger than former.

- I try to fix this problem by applying a novel loss, which use for reference the warmup tragedy, can see it in loss.py.

- It's simliar between two implements when using train set to train, but when use dev set to train, My implementation better than CasRel-pytorch-reimplement even it use a smaller pre-trained mdel. And it shows the effectiveness of my tragedy.

### Tips
- When using mutil losses, you should notice the magnitude of losses, because some big loss may devours the influence of small loss. So make those loss under same magnitude by giving a appropriate coefficient to the big loss, which is help model to fit.
- When using mutil losses, model is hard to fit, so just be patient and wait more train epochs
