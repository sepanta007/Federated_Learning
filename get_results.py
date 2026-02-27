import subprocess
import os
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import json

sub_path = '/.hydra/config.yaml'

def run_results(log_path):
    cfg = OmegaConf.load(f'{log_path}/{sub_path}')
    for client_id in range(cfg.num_clients):
        cmd = ["python3", "evaluate_model.py", "-l", log_path, '-n', str(client_id)]
        try:
            subprocess.run(cmd, check=True)
            generate_png(log_path, client_id)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for {log_path}/test_metrics/test_metrics_{client_id}.json: {e}")

                
def generate_png(log_path, client_id):
    metric_file = f'{log_path}/test_metrics/test_metrics_{client_id}.json'
    if not os.path.exists(metric_file):
        print(f"File {metric_file} does not exist. Skipping image generation.")
        return

    with open(metric_file, 'r') as f:
        metrics = json.load(f)
    
    accuracies = metrics.get('accuracies', [])
    losses = metrics.get('losses', [])
    
    fig, ax1 = plt.subplots()

    ax1.plot(accuracies, 'g-', label='Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='g')

    ax2 = ax1.twinx()
    ax2.plot(losses, 'r-', label='Loss')
    ax2.set_ylabel('Loss', color='r')

    ax1.set_title(f'Client {client_id} - Metrics')
    fig.tight_layout()

    output_path = f'{log_path}/images/client_{client_id}_metrics.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Generated PNG: {output_path}")

   
def main() -> None:
    parser = argparse.ArgumentParser(description="Get results from model")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")

    log_directory  = parser.parse_args().log_directory
    
    # we check if metric_0 exists to know if we have to run evaluation or not
    if os.path.exists(f'{log_directory}/{sub_path}') and not os.path.exists(f'{log_directory}/test_metrics/test_metrics_0.json'):
        run_results(f'{log_directory}')
        return
    for dir in os.listdir(log_directory):
        log_path = f'{log_directory}/{dir}'
        if os.path.exists(f'{log_path}/{sub_path}'):
            if not os.path.exists(f'{log_path}/test_metrics/test_metrics_0.json'):
                run_results(log_path)
        else:
            for log_dir in os.listdir(f'{log_directory}/{dir}'):
                log_path = f'{log_directory}/{dir}/{log_dir}'
                if not os.path.exists(f'{log_path}/test_metrics/test_metrics_0.json'):
                    run_results(log_path)
                
if __name__ == '__main__':
    main()
