# References and notices

[中文](../zh/references.md) | **English** · [Documentation home](../../README.md#en)

## Relationship to official Uni-Mol

`unimol_lite` is an independent third-party implementation. It is **not** the official Uni-Mol release, and it is not maintained or endorsed by the Uni-Mol authors. It keeps a lightweight path for SMILES featurization, model inference and masked-atom pretraining, intended for local experiments and secondary development.

The official implementation, pretrained weights, downstream tools and support are available upstream:

- Official repository: <https://github.com/deepmodeling/Uni-Mol>
- Official toolkit: <https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_tools>

## Citation

When using this project or derivative work, cite the original Uni-Mol paper:

```bibtex
@inproceedings{zhou2023unimol,
  title     = {Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author    = {Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=6K2RM6wVqKu}
}
```

- OpenReview: <https://openreview.net/forum?id=6K2RM6wVqKu>
- Title: *Uni-Mol: A Universal 3D Molecular Representation Learning Framework*
- Venue: ICLR 2023

## License

- `unimol_lite` is released under the **MIT License**; the full text is in [`LICENSE`](../../LICENSE) at the repository root.
- Upstream [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) is also MIT-licensed. Parts of this project's implementation and interfaces reference the upstream code and documentation.
- This project does not distribute official Uni-Mol pretrained weights or data covered by separate terms. Weights or data downloaded separately remain subject to their source terms.
