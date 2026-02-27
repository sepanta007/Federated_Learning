import sys
import os
import numpy as np
import random
import logging
from logging import INFO
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_classname import load_client_element
from flwr_datasets import FederatedDataset
from base.model import LocalOnlyModelManager
from base.utils import load_datasets
from base.partitioner import load_partitioner
from flwr_datasets.visualization import plot_label_distributions

log = logging.getLogger(__name__)

@hydra.main(config_path='../conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    log_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Saving logs to {log_save_path}")

    client_save_path = f"{log_save_path}/client_states"
    os.makedirs(client_save_path, exist_ok=True)

    train_partitioner, test_partitioner = load_partitioner(cfg)

    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        partitioners={"train": train_partitioner, "test": test_partitioner}
    )

    fig_train, _, _ = plot_label_distributions(
        partitioner=fds.partitioners["train"],
        label_name="label",
        legend=True,
    )
    fig_test, _, _ = plot_label_distributions(
        partitioner=fds.partitioners["test"],
        label_name="label",
        legend=True,
    )

    fig_train.legend()
    fig_train.set_size_inches(12, 8)
    fig_train.savefig(f'{log_save_path}/samples_per_label_per_client.png', dpi=300)
    fig_test.legend()
    fig_test.set_size_inches(12, 8)
    fig_test.savefig(f'{log_save_path}/samples_per_label_per_client_test.png', dpi=300)
    plt.close('all')

    epochs = int(
        cfg.client_config.num_epochs
        * cfg.num_rounds
        * cfg.server_config.fraction_fit
    )
    print(f"Total epochs per client: {epochs}")

    all_epoch_losses = []
    all_epoch_accuracies = []
    all_test_epoch_losses = []
    all_test_epoch_accuracies = []

    for client_id in range(cfg.num_clients):

        log.info(f"------------------ Client {client_id} ------------------")

        trainloader, _, testloader = load_datasets(client_id, fds, cfg)

        client_classname, model_manager_class, model_module_class = load_client_element(cfg)
        
        if cfg.algorithm.lower() == "local":
            model_manager_class = LocalOnlyModelManager

        model_manager = model_manager_class(
            client_id=client_id,
            config=cfg,
            trainloader=trainloader,
            testloader=testloader,
            model_class=model_module_class,
            client_save_path=f"{client_save_path}/local_net_{client_id}.pth"
        )

        client = client_classname(client_id, model_manager, cfg)
        client.epochs = epochs

        train_results = client.perform_train(verbose=True)

        all_epoch_losses.append(train_results["epoch_loss"])
        all_epoch_accuracies.append(train_results["epoch_accuracy"])
        all_test_epoch_losses.append(train_results.get("test_epoch_loss", train_results["epoch_loss"]))
        all_test_epoch_accuracies.append(train_results.get("test_epoch_accuracy", train_results["epoch_accuracy"]))

        test_results = client.perform_test(full_report=False)

        log.info(
            f"Client {client_id} | "
            f"Test Loss: {test_results['loss']:.4f} | "
            f"Test Accuracy: {test_results['accuracy']:.4f}"
        )

    avg_epoch_losses = np.mean(np.array(all_epoch_losses), axis=0)
    avg_epoch_accuracies = np.mean(np.array(all_epoch_accuracies), axis=0)
    avg_test_epoch_losses = np.mean(np.array(all_test_epoch_losses), axis=0)
    avg_test_epoch_accuracies = np.mean(np.array(all_test_epoch_accuracies), axis=0)

    avg_test_loss = np.mean([arr[-1] for arr in all_test_epoch_losses])
    avg_test_accuracy = np.mean([arr[-1] for arr in all_test_epoch_accuracies])

    print("\n================ GLOBAL RESULTS ================")
    print(f"Final Average Training Accuracy: {avg_epoch_accuracies[-1]:.4f}")
    print(f"Final Average Training Loss: {avg_epoch_losses[-1]:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print("================================================")

    history = {
        "fit": {
            "accuracy": [(i + 1, float(acc)) for i, acc in enumerate(avg_epoch_accuracies)],
            "loss": [(i + 1, float(loss)) for i, loss in enumerate(avg_epoch_losses)],
        },
        "evaluate": {
            "accuracy": [(i + 1, float(acc)) for i, acc in enumerate(avg_test_epoch_accuracies)],
            "loss": [(i + 1, float(loss)) for i, loss in enumerate(avg_test_epoch_losses)],
        },
    }

    log.info("\tHistory (metrics, distributed, fit):")
    log.info(f"\t{history['fit']}")
    log.info("\tHistory (metrics, distributed, evaluate):")
    log.info(f"\t{history['evaluate']}")

if __name__ == "__main__":
    main()
