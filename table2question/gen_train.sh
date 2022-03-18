if [ "$#" -ne 2 ]; then
  echo "Usage: ./gen_train.sh <dataset> <experiment>"
  exit
fi
dataset=$1
expr=$2
python ./gen_train_data.py --dataset ${dataset} --sql_expr auto_sql --expr ${expr}
