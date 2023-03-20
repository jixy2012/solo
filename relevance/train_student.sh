python train_student_retriever.py \
--do_train \
--lr 1e-4 \
--optim adamw \
--scheduler linear \
--teacher_model_path /home/cc/code/table_discovery_project/models/tqa_retriever \
--train_data ./data/trivia_qa/train_0.jsonl \
--eval_data ./data/trivia_qa/dev_0.jsonl \
--per_gpu_batch_size 32 \
--n_context 100 \
--total_steps 20000 \
--scheduler_steps 30000 \
--teacher_precompute_file "./data/teacher_precompute_triviaqa.pl" \
--distill_temperature 6 \
--distill_weight 0.5 \
--num_train_neg_ctxs 25 \
--num_eval_neg_ctxs 15 \
