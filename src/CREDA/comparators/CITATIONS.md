# Comparator sources

Code in this package is copied verbatim from each paper's official repository (shallow
clones fetched 2026-10-12) with the operator's explicit authorisation. Local edits are
marked `# LOCAL EDIT:` in the vendored files. PFAN is TensorFlow 1; its sources are kept
as never-imported `.txt` references and `pfan/pfan_torch.py` is a PyTorch translation.

| Method | Repository | Commit | Upstream paths | License as found |
|---|---|---|---|---|
| SHOT | https://github.com/tim-learn/SHOT | f7d555a0d53b525b885e5ef2a887a267a5be3c36 | object/loss.py, object/image_target.py | MIT |
| CDAN+E | https://github.com/thuml/CDAN | c904cd3f2fe092bb5175be1c5b6e42fa2036dece | pytorch/loss.py, pytorch/network.py | none for pytorch/ |
| DSAN (LMMD) | https://github.com/easezyc/deep-transfer-learning | 2969ed6c740d04283aa54af4e2c267cb980c96fe | UDA/pytorch1.0/DSAN/lmmd.py | MIT |
| CS-DA | https://github.com/ywzcode/CS-adv | 79d010b5e125bf4558d9318bdcd633937e0c9274 | CSutils.py | none found |
| PFAN | https://github.com/Xiewp/PFAN | 328c4c5b4e24a6d04b06136d9e83d8acb391a289 | model.py, pseudo.py, train.py | none found |
| JUMBOT | https://github.com/kilianFatras/JUMBOT | af9a22d800cf9b9c1243ab125314eefaf06fdda8 | digits/jumbot.py (code); hyperparameter presets from Domain_Adaptation/train_jumbot.py L419-425, L271-272 and digits/train.py L84-87 | MIT |
| DALN | https://github.com/xiaoachen98/DALN | e40ca55612b7d6a973cd40e50cea32e350966a88 | nwd.py, grl.py | none found |

## Local deviations from the official code

- **CDAN+E.** `RandomLayer` stores its random matrices as non-trainable buffers (they follow `.to()` and are saved in `state_dict`; same draws). The adapter picks the multilinear map by paper Eq. 8 (randomized map with d=1024 when d_f·d_g > 4096) instead of upstream's CLI default `--random False`; `use_random=` overrides.
- **JUMBOT.** Hyperparameters follow the paper's appendix C.1 (PDF p.21) as per-dataset presets: `office_home` (default; η1=0.01, η2=0.5, τ=0.5, ε=0.01), `visda` (η1=0.005, η2=1, τ=0.3, ε=0.01), `digits` (η1=0.1, η2=0.1, τ=1, ε=0.1). The `Jumbot.__init__` defaults (η1=0.001, η2=0.0001) match no reported experiment and are not used. Office-Home as the default is an assumption: the paper reports no Office-31 setting. The adapter returns the bare term (η3=1); class-stratified source minibatches (appendix C.1) are the caller's responsibility.
- **PFAN.** The discriminator is trained by its own optimiser on detached features (`discriminator_step`), with upstream's Dregloss inside its loss and 2× learning rate on biases, as in model.py; the returned generator term never writes into it.
- **SHOT.** Whole-dataset pseudo-labels are indexed with the batch's dataset indices (`tar_idx`), as upstream `mem_label[tar_idx]`.
- **DALN.** Two `# LOCAL EDIT:` lines: a relative import, and `np.float(` → `float(` (removed in NumPy ≥ 1.24). The official GRL sits between features and classifier, so the adapter re-runs the shared head passed as `head=`.

## BibTeX

```bibtex
@inproceedings{liang2020shot,
  author    = {Liang, Jian and Hu, Dapeng and Feng, Jiashi},
  title     = {Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  year      = {2020},
  note      = {arXiv:2002.08546}
}

@inproceedings{long2018cdan,
  author    = {Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I.},
  title     = {Conditional Adversarial Domain Adaptation},
  booktitle = {Advances in Neural Information Processing Systems 31 (NeurIPS)},
  year      = {2018},
  note      = {arXiv:1705.10667}
}

@article{zhu2021dsan,
  author  = {Zhu, Yongchun and Zhuang, Fuzhen and Wang, Jindong and Ke, Guolin and Chen, Jingwu and Bian, Jiang and Xiong, Hui and He, Qing},
  title   = {Deep Subdomain Adaptation Network for Image Classification},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2021},
  note    = {Early access 2020; volume/pages not verified. arXiv:2106.09388}
}

@inproceedings{yin2024csda,
  author    = {Yin, Wenzhe and Yu, Shujian and Lin, Yicong and Liu, Jie and Sonke, Jan-Jakob and Gavves, Efstratios},
  title     = {Domain Adaptation with Cauchy-Schwarz Divergence},
  booktitle = {Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2024},
  note      = {arXiv:2405.19978}
}

@inproceedings{chen2019pfan,
  author    = {Chen, Chaoqi and Xie, Weiping and Huang, Wenbing and Rong, Yu and Ding, Xinghao and Huang, Yue and Xu, Tingyang and Huang, Junzhou},
  title     = {Progressive Feature Alignment for Unsupervised Domain Adaptation},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  note      = {arXiv:1811.08585}
}

@inproceedings{fatras2021jumbot,
  author    = {Fatras, Kilian and S\'ejourn\'e, Thibault and Courty, Nicolas and Flamary, R\'emi},
  title     = {Unbalanced minibatch Optimal Transport; applications to Domain Adaptation},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year      = {2021}
}

@inproceedings{chen2022reusing,
  author    = {Chen, Lin and Chen, Huaian and Wei, Zhixiang and Jin, Xin and Tan, Xiao and Jin, Yi and Chen, Enhong},
  title     = {Reusing the Task-specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7181--7190},
  year      = {2022},
  note      = {arXiv:2204.03838; BibTeX from the official README}
}

```

Author lists for CDAN, DSAN and CS-DA were read off the arXiv PDFs' title pages; JUMBOT and
PFAN follow the BibTeX in their own READMEs; SHOT follows the citation supplied with the task
(PDF not fetched). Volume and page data are deliberately omitted (not verified).
