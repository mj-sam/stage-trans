# Stage training strategy implementation
This is the implementation code of a submited conference paper(ICBME 2021).

# Getting Started

First, clone this source code. Then, download the dataset "Four class motor imagery (001-2014)" of the BCI competition IV-2a.Put all files of the dataset (A01T.mat-A09E.mat) into a subfolder within the project called 'dataset'.

# Prerequisites
- Python == 3.7 or 3.8
- tensorflow == 2.X (verified working with 2.0 - 2.3, both for CPU and GPU)
- numpy
- sklearn
- scipy
# Run
```bash
python main.py --help


Stage and Cross training strategy

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to the dataset folder see : bcni datasets fotmat
  --patience PATIENCE   early stopping callback patience
  --epochs EPOCHS       epochs model will be trained on
  --frequency_cut_low FREQUENCY_CUT_LOW
                        lower cut-off frequency in proprocessing
  --frequency_cut_high FREQUENCY_CUT_HIGH
                        higher cut-off frequency in proprocessing
  --subject SUBJECT     target subject
  --k_fold K_FOLD
  --iterations ITERATIONS
  --fine_tune_epochs FINE_TUNE_EPOCHS
  --save_model SAVE_MODEL
                        if true cross trained model wont be deleted after
                        execution
  --stage               if true stage training will be used instead of
                        standard training
  --cross_subject       model will be pre-trained on all subjects of the data
                        set
```

-Enable stage training
```
python main.py --subject 1 --patience 50 --iterations 5 --epochs 500 --fine_tune_epochs 100 --stage --cross_subject
-------------------  ------------------
Model                EEGNet
Stage training       Enable
Cross Subjects       Enable
Training Epochs      500
Training Iterations  5
K_Fold               Enable
Accuracy             0.79679
-------------------  ------------------
```

-Disable stage training (default)
```
python main.py --subject 1 --patience 50 --iterations 5 --epochs 500 --fine_tune_epochs 100 
-------------------  ------------------
Model                EEGNet
Stage training       Disable
Cross Subjects       Disable
Training Epochs      500
Training Iterations  5
K_Fold               Enable
Accuracy             0.7523131672597865
-------------------  ------------------
```



# Authors
- Javad Sameri        sameryq@gmail.com
- Ehsan Zarooshan     hesam_zarooshan@comp.iust.ac.ir 
