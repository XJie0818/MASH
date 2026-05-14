# MASH

### Requirements
To run this model, ensure you have Python 3.9 installed.
```shell
pip install -r requirement.txt
```

## Running
To run the MASH model:
```
python -u run.py --mode=train --dataset=NYCcat --gpu=0
python -u run.py --mode=train --dataset=CAcat --gpu=0
python -u run.py --mode=train --dataset=TKYcat --hidden_size=256 --learning_rate=0.0002 --gpu=0
```

To run the robustness experiment:
```
python -u run.py --mode=train --dataset=NYCcat --drop_ratio = 0.1 --gpu=0
```

To run the user check-in volume experiment:
```
python -u run.py --mode=train --dataset=shortnyc --gpu=0
python -u run.py --mode=train --dataset=midnyc --gpu=0
python -u run.py --mode=train --dataset=highnyc --gpu=0
```
* You should adjust the max_sequence_length according to the average length of check-in data in different datasets. 

