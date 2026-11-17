# Third-party notices and data provenance

The project's original source code is available under GNU AGPL version 3. Dependencies and externally downloaded assets retain their own licenses. This notice is a provenance inventory, not a replacement for any upstream terms.

| Component | Source / terms | Local use |
|---|---|---|
| Ultralytics and YOLO11n-pose | [Repository](https://github.com/ultralytics/ultralytics), [license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [licensing options](https://www.ultralytics.com/license) | Open-source AGPL path; pretrained extraction only, no YOLO fine-tuning |
| YOLO11n-pose weights | [Official asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) | Downloaded separately; local provenance/hash retained, binary excluded from Git |
| ByteTrack | Tracker implementation provided by the installed Ultralytics package | Consult installed implementation and upstream notices |
| PyTorch / TorchVision | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE) | CPU research dependencies, installed separately |
| NumPy / SciPy / OpenCV and other libraries | Respective package licenses and notices | Exact installed versions in dependency locks; no vendored library binaries |
| PoseLift | [Publisher repository](https://github.com/TeCSAR-UNCC/PoseLift), repository Apache-2.0 license; separately hosted data terms must also be checked | Processed pose training/test files downloaded separately, not redistributed here |
| MNNIT Shoplifting Dataset V1 | [Publisher record](https://data.mendeley.com/datasets/r3yjf35hzr/1), DOI 10.17632/r3yjf35hzr.1; publisher states CC BY 4.0 | Eight unmodified video members downloaded separately; attribution preserved locally |

MNNIT citation: Ansari, Mohd. Aquib; Singh, Dushyant Kumar (2023), *Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad*, Mendeley Data, V1, doi:10.17632/r3yjf35hzr.1.

PoseLift citation: Narges Rashvand, Ghazal Alinezhad Noghre, Armin Danesh Pazho, Shanle Yao, and Hamed Tabkhi, *Exploring Pose-Based Anomaly Detection for Retail Security: A Real-World Shoplifting Dataset and Benchmark*, WACV Workshops 2025, [paper](https://arxiv.org/abs/2501.06591).

The research baseline model is the project's own small causal predictor. STG-NF implementation code was not copied or bundled. Charades footage and NVIDIA-generated event footage were not incorporated. Their restrictive/custom terms must not be treated as the project's source-code license.

Before distributing model binaries or a shop installer, complete the model/dataset artifact review and ship the applicable notices. Publishing this code is not a claim that a deployable model or installer has been cleared or produced.
