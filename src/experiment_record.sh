PROG="main" ARGS="--exp-name p-span --dataset wsc-spacy --framing P-SPAN" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name p-sent --dataset wsc-spacy --framing P-SENT" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name mc-sent-ploss --dataset wsc-spacy --framing MC-SENT-PLOSS" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name mc-sent-pair --dataset wsc-spacy --framing MC-SENT-PAIR" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name mc-sent-scale --dataset wsc-spacy --framing MC-SENT-SCALE" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name mc-sent --dataset wsc-spacy --framing MC-SENT" sbatch ~/wsc.sbatch
PROG="main" ARGS="--exp-name mc-mlm --dataset wsc-spacy --framing MC-MLM" sbatch ~/wsc.sbatch

scancel 8800296
scancel 8800297
scancel 8800298
scancel 8800299
scancel 8800300
scancel 8800301
scancel 8800302
