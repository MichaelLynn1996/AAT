
# AAT: ADAPTING AUDIO TRANSFORMER FOR VARIOUS ACOUSTICS RECOGNITION TASKS

## Introduction

This repository contains the official implementation (in PyTorch) of the **AAT: ADAPTING AUDIO TRANSFORMER FOR VARIOUS ACOUSTICS RECOGNITION TASKS** submitted to the ICASSP 2024. 

### Key Files
- The model is implemented in [./src/models/ast_models.py](./src/models/ast_models.py). You may refer to it on how to apply AAT to your model.
- The recipes are in `tasks/[esc50, speechcommands, speechcommandsv1, gtzan, openmic, urbansound8k]/run_xx.sh`, when you run `run_xx.sh`, it will call `/src/run.py`, which will then call `/src/dataloader.py` and `/src/traintest.py`, which will then call `/src/models/ast_models.py`.

## Acknowledgement
Thanks [YuanGongND](https://github.com/YuanGongND/ast) for providing such an amazing training pipline. When the paper is accepted, we will further complete this README.

## Contact
If you have a question, please bring up an issue (preferred) or send me an email sealynndev@gmail.com.

