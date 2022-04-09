if [ "$#" -ne 3 ]; then
  echo "Usage: ./gen_train_dev_retr.sh <dataset> <expr> <sql_expr>"
  exit
fi
dataset=$1
expr=$2
sql_expr=$3
python ./gen_train_dev_retr.py --dataset ${dataset} --expr ${expr} --sql_expr ${sql_expr}