This repository contains my full implementation for :

- **Part 1:** Fine-tuning **BERT** for *sentiment classification* on IMDB  
- **Part 2:** Fine-tuning **T5-small** for *Text-to-SQL* mapping on the ATIS-style flight database

Both parts include training code, evaluation scripts, model outputs, and analysis.

It is recommended to use Python **3.10+**.
```
python -m pip install -r requirements.txt
```

Debug training on a tiny dataset
```
python3 main.py --train --eval --debug train
```
Full training + evaluation on original test set
```
python3 main.py --train --eval
```
Evaluate on transformed test data
```
python3 main.py --eval transformed --debug transformation
python3 main.py --eval transformed
```
Augmented Training
```
python3 main.py --train augmented --eval transformed
```
Original test
```
python3 main.py --eval --model_dir out_augmented
```
Transformed test
```
python3 main.py --eval transformed --model_dir out_augmented
```
T5-small fine-tuned model
```
python train_t5.py \
  --finetune \
  --optimizer_type AdamW \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --scheduler_type linear \
  --num_warmup_epochs 1 \
  --max_n_epochs 50 \
  --patience_epochs 8 \
  --batch_size 16 \
  --test_batch_size 16 \
  --freeze_encoder_layers 2
```
