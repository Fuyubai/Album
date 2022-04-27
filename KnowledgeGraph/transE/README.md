## TransE
That code is a simple reproduction of [TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), and it refers to a amazing repo [OpenKE](https://github.com/thunlp/OpenKE)

Note: Just for learning, the inference performance may not stasify the production environment, though it has beed optimized 

### Quick Start
```python
python main.py
```

### Metrics
||that code|origin paper|
| :--: | :--:|:--:|
|dataset|FB15K237|FB15K|
|Mean Rank|0.3060|0.3490|
|Hits@10|370|243|
