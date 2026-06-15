
# WandB Report

ექსპერიმენტების დასრულების შემდეგ wandb.ai-ზე შევქმენი report სადაც ყველა run-ის შედეგები შევაჯამე.

---

## რა ჩავრთე Report-ში

### 1. პროექტის მიმოხილვა

სულ 9 ექსპერიმენტი გავუშვი 4 სხვადასხვა არქიტექტურაზე. მიზანი იყო პატარა baseline-იდან დაწყება და თანდათანობით გართულება — ისე რომ ნათლად ჩანდეს რატომ ვამატებდი თითოეულ ცვლილებას.

---

### 2. Underfitting — exp01, exp02

პირველ ორ run-ში TinyFERNet გამოვიყენე. val/acc დაახლოებით 40-42%-ზე გაჩერდა და მეტი ვერ ასწია, augmentation-ის დამატების შემდეგაც კი. ეს ნიშნავს რომ მოდელს არ ეყოფა capacity — ძალიან მარტივია ამოცანისთვის.

WandB-ში ავაგე line plot: `val/acc` vs `epoch`, run-ები: exp01 და exp02.

---

### 3. Overfitting — exp03

MediumFERNet-ი dropout-ისა და augmentation-ის გარეშე გავუშვი. train accuracy ~52%-მდე ავიდა მაგრამ val accuracy ~46%-ზე დარჩა. gap epoch 15-ის შემდეგ გახდა თვალსაჩინო.

ავაგე ორი panel:
- `train/acc` vs `val/acc` — ერთ გრაფიკზე, ნათლად ჩანს გამოყოფა
- `train_val_acc_gap` — ცალკე, gap-ის ზრდა epoch-ების მიხედვით

---

### 4. რეგულარიზაციის ეფექტი — exp03 vs exp04

exp04-ში დავამატე dropout(0.5), weight decay და augmentation. gap-ი მნიშვნელოვნად შემცირდა. ამ ორი run-ის შედარება კარგად აჩვენებს რომ regularization მართლა მუშაობს.

Panel: `val/acc` და `train_val_acc_gap` — exp03 vs exp04 ერთ გრაფიკზე.

---

### 5. Optimizer-ების შედარება — exp04 vs exp05

exp04 AdamW-ით, exp05 SGD-ით (lr=0.01, momentum=0.9). Adam-ი პირველ epoch-ებში უფრო სწრაფად კონვერგირდება, SGD-ი კი LR decay-ის შემდეგ ეწევა. საბოლოო val accuracy-ი მსგავსია.

Panel: `val/acc` vs `epoch` — ორივე run ერთად.

---

### 6. DeepFERNet და class weights — exp06 vs exp07

exp06 უბრალოდ უფრო ღრმა ქსელია. exp07-ში Disgust კლასს მეტი წონა მივეცი (x5) რადგან dataset-ში ყველაზე ნაკლებია (~436 სურათი). overall accuracy ოდნავ დაეცა მაგრამ Disgust-ის F1 score მნიშვნელოვნად გაიზარდა.

ჩავამატე per-class F1 score bar chart — `class/Disgust/f1` exp06 vs exp07.

---

### 7. Transfer Learning — exp08 vs exp09

exp08-ში ResNet18-ის backbone გავაყინე და მხოლოდ FC layer ვასწავლე. exp09-ში სრული fine-tuning გავაკეთე lr=1e-4-ით. სხვაობა საკმაოდ დიდია — fine-tuning-ი საუკეთესო შედეგს იძლევა.

Panel: `val/acc` vs `epoch` — exp08 vs exp09.

---

### 8. საბოლოო შედარება

ბოლოს ავაგე bar chart სადაც ყველა 9 run-ის `best_val_acc` ჩანს. ნათლად ჩანს პროგრესია tiny → medium → deep → resnet18.

ასევე ჩავამატე თითოეული მოდელის confusion matrix — საინტერესოა რომ Disgust და Fear ყველა მოდელისთვის ყველაზე რთული კლასებია.

---

## Report-ის ლინკი

https://wandb.ai/adane21-free-university-of-tbilisi-/fer2013-experiments/reports/Untitled-Report--VmlldzoxNzIyNTY4Ng/edit?draftId=VmlldzoxNzIyNTY4Ng%3D%3D&importPanel=
