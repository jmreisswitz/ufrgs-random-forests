# ufrgs-random-forests

## Installing dependencies
```shell script
pip3 install -r requirements.txt
```

## Running
```shell script
python3 main.py --ntree <ntree> --target <target_column> --dataset <dataset path>
```
or simple
```shell script
python3 main.py
```
for default parameters

## Running benchmark dataset
Simple
```shell script
python3 benchmark.py
```

## Generating results .csvs
Simple run 
```shell script
./run_all.sh
```
