"""
Hyper parameter tuning based on RoBERTa on WSC
"""

import json
import subprocess
from random import randrange, random
import os
import argparse
import logging as log
from datetime import datetime as dt
import pandas as pd

lr_candidates = [1e-5, 2e-5, 3e-5]
batch_candidates = [8, 16, 32, 64]
updates_candidates = [500, 1000, 2000, 3000]
seed_range = 1e6

def run_trials(args):
    results = {
        "trial"         : [],
        "learning rate" : [],
        "batch size"    : [],
        "n updates"     : [],
        "seed"          : [],
        "best accuracy" : []
        }
    
    for trial in range(args.n_trials):
        # select candidates for trial
        lr, batch, updates, seed = select_candidates()
        
        command = [
            'python',
            os.path.join('.','wsc-trick','src','main.py'),
            f'--exp-name={args.study_name}',
            f'--dataset={args.dataset}',
            f'--framing={args.framing}',
            f'--bs={batch}',
            f'--lr={lr}',
            f'--weight-decay={args.weight_decay}',
            f'--max-epochs={updates}',
            f'--warmup-iters-ratio={args.warmup_iters_ratio}',
            f'--results-dir={args.results_dir}',
            f'--data-dir={args.data_dir}'
            ]
        
        if batch > 8:
            command.append('--amp')
        
        command = [
            'python',
            os.path.join('.','wsc-trick','src','main.py'),
            '--reload-data',
            f'--exp-name={args.study_name}',
            f'--dataset={args.dataset}',
            f'--framing={args.framing}',
            f'--results-dir={args.results_dir}',
            f'--data-dir={args.data_dir}'
            ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()
        
        record_trial(results, trial, lr, batch, updates, seed)
        
    with open(os.path.join(args.results_dir, 'val_summary.jsonl'), 'a') as reader:
        for row in reader:
            result = json.loads(row)
            results['best accuracy'].append(result['best_acc'])
    
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.results_dir, '{}_results.csv'.format(args.study_name)),
                   index=False)
    
def record_trial(results,
                  trial,
                  lr,
                  batch,
                  updates,
                  seed):
    results['trial'].append(trial)
    results['learning rate'].append(lr)
    results['batch size'].append(batch)
    results['n updates'].append(updates)
    results['seed'].append(seed)

def select_candidates():
    lr = lr_candidates[randrange(0,len(lr_candidates),1)]
    batch = batch_candidates[randrange(0,len(batch_candidates),1)]
    updates = updates_candidates[randrange(0,len(updates_candidates),1)]
    seed = int(random()*seed_range)
    
    return lr, batch, updates, seed

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    parser = argparse.ArgumentParser(description="Run Optuna trails")
    parser.add_argument("--study-name", type=str, help="experiment name")
    parser.add_argument("--n-trials", type=int, help="number of trials")
    parser.add_argument("--results-dir", type=str, help="results directory")
    parser.add_argument("--data-dir", type=str, help="stored data")
    parser.add_argument("--dataset",
                        type=str,
                        default="wsc-cross",
                        choices=[
                            "wsc-spacy",
                            "wsc-cross",
                            "winogrande-xs",
                            "winogrande-s",
                            "winogrande-m",
                            "winogrande-l",
                            "winogrande-xl",])
    
    parser.add_argument("--framing",
                        type=str,
                        choices=[
                            "P-SPAN",
                            "P-SENT",
                            "MC-SENT-PLOSS",
                            "MC-SENT-PAIR",
                            "MC-SENT-SCALE",
                            "MC-SENT",
                            "MC-MLM",])
    
    # training settings
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-iters-ratio", type=float, default=0.06)
    
    args = parser.parse_args()
    
    log_fname = os.path.join(file_dir, "{}_log_{}.log".format(
        args.study_name,
        dt.now().strftime("%Y%m%d_%H%M")))
    log.basicConfig(filename=log_fname,
        format='%(asctime)s - %(name)s - %(message)s',
        level=log.INFO)
    
    run_trials(args)