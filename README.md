<div align="center">

# TabDPT: Scaling Tabular Foundation Models on Real Data

[![arxiv](https://img.shields.io/static/v1?label=arXiv&message=2410.18164&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2410.18164)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Layer6/TabDPT)

</div>

**TabDPT** is an open-source foundation model for tabular data based on in-context learning (ICL). It is trained on real-world data and can generalize to new tasks **without** additional training or hyperparameter tuning.

This repository provides lightweight interface code to generate predictions on new tabular datasets. Full training code is available [here](https://github.com/layer6ai-labs/TabDPT-training/).

## Usage

### Installation

TabDPT is available on [Hugging Face](https://huggingface.co/Layer6/TabDPT).

To set up this repo, first ensure you have Python 3.10 or 3.11. Then, run the following:
```
git clone git@github.com:layer6ai-labs/TabDPT.git
cd TabDPT
pip install -e .
pip install --group dev
```

Alternatively, if you are using a package manager such as `uv`, you can run
```
uv sync
```

You may also need a C++ compiler such as `g++` for building dependencies. On Ubuntu, you can install it with:
```
sudo apt-get update
sudo apt-get install g++
```

Lastly, you will need a `python-dev` system package. On Ubuntu, it can be installed with:
```
sudo apt-get update
sudo apt-get install python-dev
```

### Tips

If you experience errors caused by `torch compile` (e.g., `InductorError`), try updating package versions and system drivers. 

For better runtime performance, adjust `context_size` or `n_ensembles` to trade off speed and accuracy.

### Working Examples

See `tests/cls_example.py` and `tests/reg_example.py` for examples of how to use TabDPT once installed.

## Overview

TabDPT uses retrieval and self-supervised learning to remove constraints on dataset size and to enable effective generalization from pre-training on real data. We find this to be competitive with existing ICL training approaches, and outperform leading deep learning and tree-based models:

<table align="center">
	<colgroup>
		<col />
		<col />
		<col />
		<col />
		<col />
	</colgroup>
	<thead>
		<tr class="header">
			<td style="text-align: center;" rowspan=2><strong>Algorithm</strong></th>
			<th style="text-align: center;" colspan=2><strong>CC18</strong></th>
			<th style="text-align: center;" colspan=2><strong>CTR23</strong></th>
		</tr>
		<tr class="header">
			<th style="text-align: center;"><strong>AUC</strong></th>
			<th style="text-align: center;"><strong>Accuracy</strong></th>
			<th style="text-align: center;"><strong>Correlation</strong></th>
			<th style="text-align: center;"><strong><span class="math inline"><em>R</em><sup>2</sup></span></strong></th>
		</tr>
	</thead>
	<tbody>
		<tr class="odd">
			<td style="text-align: center;"><strong>TabDPT v1.1 (Ours)</strong></td>
			<td><strong>0.976 <sub><sup>[0.974, 0.978]</sup></sub></strong></td>
			<td><strong>0.928 <sub><sup>[0.926, 0.931]</sup></sub></strong></td>
			<td><strong>0.920 <sub><sup>[0.918, 0.922]</sup></sub></strong></td>
			<td><strong>0.847 <sub><sup>[0.843, 0.851]</sup></sub></strong></td>
		</tr>
		<tr class="even">
			<td style="text-align: center;">TabDPT v1.0 (Ours)</td>
			<td>0.972 <sub><sup>[0.971, 0.973]</sup></sub></td>
			<td>0.917 <sub><sup>[0.915, 0.919]</sup></sub></td>
			<td>0.911 <sub><sup>[0.908, 0.913]</sup></sub></td>
			<td>0.831 <sub><sup>[0.826, 0.835]</sup></sub></td>
		</tr>
		<tr class="odd">
			<td style="text-align: center;"><a href="https://www.nature.com/articles/s41586-024-08328-6">TabPFN v2</a></td>
			<td>0.972 <sub><sup>[0.970, 0.974]</sup></sub></td>
			<td>0.917 <sub><sup>[0.915, 0.919]</sup></sub></td>
			<td>0.917 <sub><sup>[0.911, 0.921]</sup></sub></td>
			<td>0.841 <sub><sup>[0.831, 0.848]</sup></sub></td>
		</tr>
		<!---
		<tr class="even">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/2406.05207">TabPFN (kNN)</a></td>
			<td>0.959 <sub><sup>[0.956, 0.962]</sup></sub></td>
			<td>0.884 <sub><sup>[0.881, 0.887]</sup></sub></td>
			<td style="text-align: center;">N/A</td>
			<td style="text-align: center;">N/A</td>
		</tr>
		--->
		<tr class="odd">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/2207.01848">TabPFN</a></td>
			<td>0.939 <sub><sup>[0.935, 0.943]</sup></sub></td>
			<td>0.852 <sub><sup>[0.849, 0.856]</sup></sub></td>
			<td style="text-align: center;">N/A</td>
			<td style="text-align: center;">N/A</td>
		</tr>
		<tr class="even">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/2307.14338">TabR</a></td>
			<td>0.967 <sub><sup>[0.965, 0.969]</sup></sub></td>
			<td>0.923 <sub><sup>[0.920, 0.926]</sup></sub></td>
			<td>0.909 <sub><sup>[0.905, 0.912]</sup></sub></td>
			<td>0.825 <sub><sup>[0.817, 0.831]</sup></sub></td>
		</tr>
		<tr class="odd">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/2203.05556">MLP-PLR</a></td>
			<td>0.967 <sub><sup>[0.965, 0.968]</sup></sub></td>
			<td>0.914 <sub><sup>[0.911, 0.917]</sup></sub></td>
			<td>0.907 <sub><sup>[0.904, 0.910]</sup></sub></td>
			<td>0.827 <sub><sup>[0.822, 0.832]</sup></sub></td>
		</tr>
		<tr class="even">
			<td style="text-align: center;">MLP</td>
			<td>0.915 <sub><sup>[0.909, 0.920]</sup></sub></td>
			<td>0.865 <sub><sup>[0.860, 0.870]</sup></sub></td>
			<td style="text-align: center;">N/A</td>
			<td style="text-align: center;">N/A</td>
		</tr>
		<tr class="odd">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/1603.02754">XGBoost</a></td>
			<td>0.965 <sub><sup>[0.963, 0.967]</sup></sub></td>
			<td>0.910 <sub><sup>[0.906, 0.913]</sup></sub></td>
			<td>0.904 <sub><sup>[0.900, 0.907]</sup></sub></td>
			<td>0.820 <sub><sup>[0.814, 0.825]</sup></sub></td>
		</tr>
		<tr class="even">
			<td style="text-align: center;"><a href="https://github.com/microsoft/LightGBM">LightGBM</a></td>
			<td>0.964 <sub><sup>[0.962, 0.967]</sup></sub></td>
			<td>0.906 <sub><sup>[0.902, 0.909]</sup></sub></td>
			<td>0.900 <sub><sup>[0.896, 0.904]</sup></sub></td>
			<td>0.809 <sub><sup>[0.803, 0.815]</sup></sub></td>
		</tr>
		<tr class="odd">
			<td style="text-align: center;"><a href="https://arxiv.org/abs/1810.11363">CatBoost</a></td>
			<td>0.964 <sub><sup>[0.962, 0.967]</sup></sub></td>
			<td>0.908 <sub><sup>[0.905, 0.910]</sup></sub></td>
			<td>0.897 <sub><sup>[0.890, 0.903]</sup></sub></td>
			<td>0.802 <sub><sup>[0.794, 0.810]</sup></sub></td>
		</tr>
	</tbody>
</table>
<p>
	<b>Table 1:</b> Model performance comparison on the <a href="https://new.openml.org/search?type=study&study_type=task&id=99">CC18</a> and <a href="https://www.openml.org/search?type=study&study_type=task&id=353">CTR23</a> benchmarks, with 95% confidence intervals. Tree-based models are taken from <a href="https://arxiv.org/abs/2305.02997">McElfresh et al.</a>, MLP-PLR and TabR are taken from the <a href="https://github.com/yandex-research/tabular-dl-tabr">official implementation</a>. TabPFN-v2 was run with the default setting <code>SUBSAMPLE_SAMPLES = 10000</code> for inference. TabDPT has context size 2048 and ensemble size 8.
</p>

TabDPT is trained on real-world tabular data and we observe scaling laws similar to LLMs opening the door to training Internet-scale tabular foundation models:

<p align="center">
<img
src="https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/main/figures/scaling.png" width="50%">
<br />
<p>
<b>Figure 1:</b> Increasing model or pre-training data size (number of cells) leads to consistent improvements predictable by power laws (fitted solid lines).
</p>
</p>

TabDPT also stands out in head-to-head model comparisons and is significantly faster than other models in total time taken to generate a prediction:

<p align="center">
<img
src="https://github.com/layer6ai-labs/TabDPT-inference/raw/main/figures/performance-comparison.png" width="100%">
<br />
<p>
<b>Figure 2:</b> (<i>left</i>) Pairwise win-rate comparison in terms of classification/regression accuracy/R<sup>2</sup>. (<i>right</i>) Total runtime vs performance. TabDPT models are ordered by context size.
</p>
</p>

For full details, please see our paper [*TabDPT: Scaling Tabular Foundation Models on Real Data*](https://arxiv.org/abs/2410.18164).

## Reproducing TabDPT Paper Numbers

It is impossible to exactly replicate the results of TabDPT between runs, but this section describes how to generate results using the same evaluation approach as in the paper.

To install the dependency versions used in the paper, run
```
pip install .[reproduce-results]
```
This requires Python 3.11.

Running the `paper_evaluation.py` script will enable calculation of results similar to the paper. Run the following two commands:
```
python paper_evaluation.py --fold 0
python paper_evaluation.py --fold 1
```
and then use `notebooks/analysis.ipynb` with the resulting CSV outputs to reproduce the numbers (including confidence intervals) from the Appendix.

You can do something similar to get the ranked table from this README and the paper but will need to also compute the baseline results beforehand (code not provided in this repository).


## Citation
```
@article{ma2024tabdpt,
  title={TabDPT: Scaling Tabular Foundation Models on Real Data},
  author={Ma, Junwei and Thomas, Valentin and Hosseinzadeh, Rasa and Kamkari, Hamidreza and Labach, Alex and Cresswell, Jesse C and Golestan, Keyvan and Yu, Guangwei and Caterini, Anthony L and Volkovs, Maksims},
  journal={arXiv preprint arXiv:2410.18164},
  year={2024}
}
```

© Copyright 2024-2025 The Toronto-Dominion Bank and/or its affiliates
