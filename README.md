# FER2013 – სახის გამომეტყველების ამოცნობა

**Kaggle Competition:** [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

**WandB პროექტი:** [fer2013-experiments](https://wandb.ai/adane21-free-university-of-tbilisi-/fer2013-experiments)

**WandB Report:** https://wandb.ai/adane21-free-university-of-tbilisi-/fer2013-experiments/reports/FER2013-Experiment-Analysis--VmlldzoxNzIyNTY4Ng

**Google Colab Notebook:** `notebooks/fer2013_experiments.ipynb`
## ამოცანის აღწერა

FER2013 dataset-ი შეიცავს 48×48 პიქსელის ნაცრისფერ სახის სურათებს, რომლებიც 7 ემოციურ კლასად არის დაყოფილი. ამოცანაა სწორი კლასის პრედიქცია.

| ემოცია | ლეიბლი | სურათების რ-ბა |
|--------|--------|---------------|
| Angry (გაბრაზებული) | 0 | 3,995 |
| Disgust (ზიზღი) | 1 | 436 |
| Fear (შიში) | 2 | 4,097 |
| Happy (ბედნიერი) | 3 | 7,215 |
| Sad (მოწყენილი) | 4 | 4,830 |
| Surprise (გაკვირვებული) | 5 | 3,171 |
| Neutral (ნეიტრალური) | 6 | 4,965 |

**მთავარი სირთულე:** კლასების დისბალანსი — Disgust კლასი 16× ნაკლებია Happy-სთან შედარებით.

**Dataset splits:**
- Training: 28,709 სურათი
- PublicTest: 3,589 სურათი
- PrivateTest: 3,589 სურათი

---

## რეპოზიტორიის სტრუქტურა

```
fer2013-experiments/
├── src/
│   └── train.py                    # ყველა მოდელი + train loop + WandB logging
├── notebooks/
│   └── fer2013_experiments.ipynb   # Colab notebook (GPU-ზე გაშვება)
├── configs/                        # ექსპერიმენტების კონფიგი
├── results/                        # confusion matrix-ები, გრაფები
├── submission.csv                  # Kaggle submission (საუკეთესო მოდელი)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Sanity Checks

ყოველი ექსპერიმენტის დაწყებამდე 3 დიაგნოსტიკური ტესტი სრულდება:

**1. Forward Check** — ერთი batch გადის ქსელში, მოწმდება output-ის ზომა და მნიშვნელობები.

**2. Backward Check** — მოწმდება gradient-ების გავრცელება. თუ gradient-ები 0-ია — მოდელი ვერ ისწავლის.

**3. Single Batch Overfit** — მოდელი 100 iteration-ს ტრენინგს გადის 8 სურათზე. თუ ვერ ახდენს overfit-ს — ეს ნიშნავს, რომ არქიტექტურას არ აქვს საკმარისი capacity.

WandB-ზე ილოგება `sanity/overfit_loss` და `sanity/overfit_acc` სახელით.

---

## ექსპერიმენტების აღწერა

### ექსპ. 01 — `TinyFERNet` (baseline, augmentation გარეშე)

```
Conv(3→16) → ReLU → MaxPool →
Conv(16→32) → ReLU → MaxPool →
FC(32×12×12 → 128) → ReLU → FC(128→7)
```

**გადაწყვეტილება:** დავიწყოთ ყველაზე მარტივი არქიტექტურით — საბაზისო შედეგის დასადგენად.
**მოსალოდნელი შედეგი:** **Underfitting** — მოდელს არ ყოფნის capacity ემოციების სიღრმისეული ნიშნების ამოსაცნობად.
**რას ვსწავლობთ:** რა დონის სიზუსტე მიიღება მინიმალური capacity-ით.

---

### ექსპ. 02 — `TinyFERNet` + Augmentation

იგივე არქიტექტურა, დამატებულია: random horizontal flip, ±10° rotation, color jitter.

**გადაწყვეტილება:** Augmentation-ი მცირე მოდელსაც ეხმარება გენერალიზაციაში.
**მოსალოდნელი შედეგი:** Exp01-ზე მცირედი გაუმჯობესება, მაგრამ underfitting კვლავ შენარჩუნებულია.
**რას ვსწავლობთ:** რამდენი შეიძლება მოიგოს მხოლოდ data augmentation-მა capacity-ის ზრდის გარეშე.

---

### ექსპ. 03 — `MediumFERNet` (რეგულარიზაციის გარეშე)

```
Block(3→32) → Block(32→64) → Block(64→128) →
FC(128×6×6 → 512 → 256 → 7)
# ყოველი Block: Conv-BN-ReLU × 2 → MaxPool
# dropout = 0.0 ყველგან, augmentation = False
```

**გადაწყვეტილება:** გავზარდოთ capacity, მაგრამ განზრახ ვამოღოთ ყველა რეგულარიზაცია.
**მოსალოდნელი შედეგი:** **Overfitting** — train accuracy >> val accuracy ~20 epoch-ის შემდეგ. `train_val_acc_gap > 0.10`.
**რას ვსწავლობთ:** Capacity-ის ზრდა რეგულარიზაციის გარეშე საზიანოა.

---

### ექსპ. 04 — `MediumFERNet` + Dropout(0.5) + AdamW + Augmentation

იგივე არქიტექტურა: dropout=0.5, weight_decay=1e-3, augmentation=True, optimizer=AdamW.

**გადაწყვეტილება:** Exp03-ის overfitting-ი გამოვასწოროთ სტანდარტული ინსტრუმენტებით.
**მოსალოდნელი შედეგი:** Train/val gap მცირდება, val accuracy იზრდება.
**რას ვსწავლობთ:** Dropout + weight decay + augmentation — ეს სამი ინსტრუმენტი ეფექტური კომბინაციაა.

---

### ექსპ. 05 — `MediumFERNet` + SGD (lr=0.01, momentum=0.9, step decay)

Exp04-ის იგივე კონფიგი, მხოლოდ optimizer შეცვლილია: SGD + StepLR scheduler, batch_size=128.

**გადაწყვეტილება:** Adam vs SGD — რომელი optimizer მუშაობს უკეთ ამ dataset-ზე?
**ჰიპოთეზა:** Adam უფრო სწრაფად კონვერგირდება, SGD-ს კი უფრო კარგი generalization შეიძლება ჰქონდეს.
**რას ვსწავლობთ:** Optimizer-ის გავლენა კონვერგენციის სიჩქარეზე და საბოლოო სიზუსტეზე.

---

### ექსპ. 06 — `DeepFERNet`

```
Block(3→64) → Block(64→128) → Block(128→256) → Block(256→256) →
AdaptiveAvgPool(3×3) → FC(2304 → 1024 → 512 → 7)
dropout=0.4, optimizer=AdamW, aug=True
```

**გადაწყვეტილება:** კიდევ ერთი layer-ის დამატება, AdaptiveAvgPool — resolution-independent.
**მოსალოდნელი შედეგი:** Exp04-ზე უკეთესი, შესაძლოა მცირე overfitting 28K dataset-ზე.
**რას ვსწავლობთ:** Depth vs regularization trade-off.

---

### ექსპ. 07 — `DeepFERNet` + Class-Weighted Loss + ReduceLROnPlateau

Exp06-ის იგივე, მაგრამ CrossEntropyLoss-ი ავწონოთ: Disgust × 5.0, Happy × 0.5.

**გადაწყვეტილება:** Disgust კლასი dataset-ის მხოლოდ 1.5%-ია — მოდელი ამ კლასს იგნორირებს.
**მოსალოდნელი შედეგი:** Disgust-ის F1 score მნიშვნელოვნად გაუმჯობესება, overall accuracy შეიძლება ოდნავ დაეცეს.
**რას ვსწავლობთ:** Weighted loss-ის გავლენა imbalanced dataset-ზე.

---

### ექსპ. 08 — `ResNet18` (Frozen Backbone — Linear Probe)

ImageNet-ზე წინასწარ გაწვრთნილი ResNet18, backbone გაყინულია, მხოლოდ FC layer-ი ისწავლება.

**გადაწყვეტილება:** ImageNet feature-ები გადაეცემა თუ არა ნაცრისფერ ემოციურ სურათებს?
**მოსალოდნელი შედეგი:** სწრაფი კონვერგენცია, მაგრამ შეზღუდული ceiling — frozen feature-ები სპეციფიური დომენისთვის ოპტიმალური არ არის.
**რას ვსწავლობთ:** Transfer learning — linear probe-ის ლიმიტები ამ დომენში.

---

### ექსპ. 09 — `ResNet18` (Full Fine-Tuning)

Exp08-ის იგივე, backbone-ი გახსნილია, lr=1e-4 (პატარა — pretrained წონების შენარჩუნებისთვის), batch=32.

**გადაწყვეტილება:** სრული fine-tuning ყველაზე მაღალ სიზუსტეს მისცემს.
**მოსალოდნელი შედეგი:** **საუკეთესო val accuracy** ყველა ექსპერიმენტს შორის.
**რას ვსწავლობთ:** Fine-tuning dynamics — რამდენი epoch სჭირდება კარგ კონვერგენციას.

---

## Hyperparameter-ების შედარება

| ექს. | არქიტექტურა | LR | Optimizer | Dropout | Aug | მოსალოდნელი |
|-----|------------|-----|-----------|---------|-----|------------|
| 01 | Tiny | 1e-3 | Adam | 0 | ✗ | Underfit |
| 02 | Tiny | 1e-3 | Adam | 0 | ✓ | Underfit (ნაკლები) |
| 03 | Medium | 1e-3 | Adam | 0 | ✗ | Overfit |
| 04 | Medium | 5e-4 | AdamW | 0.5 | ✓ | — |
| 05 | Medium | 0.01 | SGD | 0.4 | ✓ | LR sensitivity |
| 06 | Deep | 3e-4 | AdamW | 0.4 | ✓ | — |
| 07 | Deep | 3e-4 | AdamW | 0.4 | ✓ | Class imbalance |
| 08 | ResNet18 frozen | 1e-3 | Adam | 0.5 | ✓ | Transfer ლიმიტი |
| 09 | ResNet18 full | 1e-4 | AdamW | 0.5 | ✓ | **საუკეთესო** |

---

## WandB Tracking სტრუქტურა

ყოველ run-ში შემდეგი მეტრიკები ილოგება:

| მეტრიკა | აღწერა |
|---------|--------|
| `train/loss`, `train/acc` | Epoch-ური სასწავლო მეტრიკები |
| `val/loss`, `val/acc` | Epoch-ური ვალიდაციის მეტრიკები |
| `train_val_loss_gap` | Generalization gap (დადებითი = overfitting) |
| `train_val_acc_gap` | Accuracy-ს generalization gap |
| `lr` | Learning rate ყოველ epoch-ზე |
| `sanity/overfit_loss` | Single-batch overfit დიაგნოსტიკა |
| `sanity/overfit_acc` | Single-batch overfit დიაგნოსტიკა |
| `confusion_matrix` | კლასთაშორისი confusion matrix (სურათი) |
| `class/{name}/f1` | თითოეული კლასის F1 score |
| `class/{name}/precision` | თითოეული კლასის precision |
| `class/{name}/recall` | თითოეული კლასის recall |

---


## შედეგები

| ექსპერიმენტი | Val Accuracy | შენიშვნა |
|-------------|-------------|---------|
| exp01_tiny_baseline | 54.0% | Underfitting დადასტურდა |
| exp02_tiny_augmented | 57.5% | მინიმალური გაუმჯობესება |
| exp03_medium_no_reg | 62.8% | Overfitting დადასტურდა |
| exp04_medium_regularized | 61.3% | რეგულარიზაცია ეფექტურია |
| exp05_medium_sgd | 62.9% | Adam-ზე კონკურენტული |
| exp06_deep_cnn | 67.9% | სიღრმე ეხმარება |
| exp07_deep_class_weights | 67.1% | Disgust F1 გაიზარდა |
| exp08_resnet18_frozen | 34.6% | Transfer learning-ის ლიმიტი — CPU-ზე გაეშვა |
| exp09_resnet18_finetune | **66.9%** | **საუკეთესო** |

---

## მთავარი დასკვნები

1. **Underfitting (Exp01-02):** TinyFERNet 54-57%-ზე სტაბილიზირდება — capacity არის ბოთლნეკი.
2. **Overfitting (Exp03):** MediumFERNet რეგულარიზაციის გარეშე — train/val gap იზრდება epoch-ების მატებასთან ერთად.
3. **რეგულარიზაცია (Exp04):** Dropout + weight_decay + augmentation კომბინაცია ეფექტურია.
4. **SGD vs Adam (Exp05):** შედეგი მსგავსია — 62.9% vs 61.3%.
5. **სიღრმე (Exp06):** DeepFERNet საუკეთესო შედეგს იძლევა custom არქიტექტურებს შორის — 67.9%.
6. **Transfer Learning (Exp08):** Frozen backbone მხოლოდ 34.6% — ImageNet feature-ები საკმარისი არ არის.
7. **Fine-tuning (Exp09):** ResNet18 სრული fine-tuning-ით 66.9% — transfer learning-ი მუშაობს მხოლოდ fine-tuning-ით.
