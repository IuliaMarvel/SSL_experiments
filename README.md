# SSL_experiments

Code for training SSL models.

In notebooks SSL(standard training).ipynb and SSL(finetuning).ipynb there is a general pipeline for training procedure.
For evaluation KNN classification (cross-validation score) on extracted features is used (see knn_utils.py).

Standard training follows [Lightly](https://github.com/lightly-ai/lightly) framework, while in finetuning pretrained weights are obtained from [MMPreTrain](https://github.com/open-mmlab/mmpretrain/tree/main#installation) (see model_utils.py for exact code of the models).
