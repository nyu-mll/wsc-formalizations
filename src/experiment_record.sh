PROG="main" ARGS="--reload-data --exp-name p-span --dataset wsc-cross --framing P-SPAN" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name p-sent --dataset wsc-cross --framing P-SENT" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name mc-sent-ploss --dataset wsc-cross --framing MC-SENT-PLOSS" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name mc-sent-pair --dataset wsc-cross --framing MC-SENT-PAIR" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name mc-sent-scale --dataset wsc-cross --framing MC-SENT-SCALE" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name mc-sent --dataset wsc-cross --framing MC-SENT" sbatch ~/wsc.sbatch
PROG="main" ARGS="--reload-data --exp-name mc-mlm --dataset wsc-cross --framing MC-MLM" sbatch ~/wsc.sbatch
