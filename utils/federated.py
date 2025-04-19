import ray
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from . import tools, models

@ray.remote # Dekoriert die Klasse als Ray Actor
class ClientActor:
    def __init__(self,
                 client_id: int,
                 X_train: Any,
                 y_train: Any,
                 X_val: Any,
                 y_val: Any,
                 config: Dict[str, Any],
                 hyperparameters: Dict[str, Any]):
        self.client_id = client_id
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.n_samples_train = tools.get_feature_dim(self.X_train)
        self.n_samples_val = tools.get_feature_dim(self.X_val)
        self.config = config
        self.hyperparameters = hyperparameters
        # Wichtig: Das Modell muss innerhalb des Actors erstellt werden
        self.model = models.get_model(config=self.config,
                                      hyperparameters=self.hyperparameters)
        self.logger = logging.getLogger(f"ClientActor_{self.client_id}")
        self.logger.setLevel(logging.INFO)
        logging.info(f"[ClientActor {self.client_id}] Initialized model.")

    def train(self,
              global_weights: List):
        """Performs model training for a communication round."""
        verbose = self.config['fl']['verbose']
        if self.X_val is not None and self.y_val is not None:
            validation_data = (self.X_val, self.y_val)
        else:
            validation_data = None
        # if config['model']['callbacks']:
        #     callbacks = [keras.callbacks.ModelCheckpoint(f'models/{config["model_name"]}.keras', save_best_only=True)]
        self.model.set_weights(global_weights)
        history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            validation_data=validation_data,
            #callbacks = callbacks if config['model']['callbacks'] else None,
            shuffle=self.config['model']['shuffle'],
            verbose=self.config['model']['verbose'],
        )
        if verbose:
            logging.info(f"[ClientActor {self.client_id}] Terminated fitting.")
        metrics = {}
        for key, value in history.history.items():
            metrics[key] = value[-1] # get metric in last round
        results = {'client_id': self.client_id,
                   'weights': self.model.get_weights(),
                   'n_samples': self.n_samples_train,
                   'metrics': metrics,
                   'history': history}
        return results

    def evaluate(self,
                 global_weights: list):
        """Performs local evaluation at the end of a communication round."""
        self.model.set_weights(global_weights)
        if self.X_val is not None and self.y_val is not None:
            metrics = self.model.evaluate(
                x=self.X_val,
                y=self.y_val,
                batch_size=self.hyperparameters['batch_size'],
                verbose=self.config['model']['verbose'],
                return_dict=True
            )
        else:
            logging.warning(f"[ClientActor {self.client_id}] No validation data available for evaluation.")
            metrics = None
        logging.info(f"[ClientActor {self.client_id}] Terminated local evaluation.")
        results = {'client_id': self.client_id,
                   'n_samples': self.n_samples_val,
                   'metrics': metrics}
        return results


def aggregate_weights(client_results: List[Dict[str, Any]]):
    total_samples = sum(client['n_samples'] for client in client_results)
    weights_agg = [np.zeros_like(layer_weights) for layer_weights in client_results[0]['weights']]
    for client in client_results:
        client_weights = client['weights'] if 'weights' in client else None
        num_samples = client['n_samples']
        weight_factor = num_samples / total_samples
        if client_weights:
            for i, layer_weights in enumerate(client_weights):
                weights_agg[i] += np.array(layer_weights) * weight_factor
        else:
            client_weights = None
    return weights_agg

def aggregate_metrics(client_results: List[Dict[str, Any]]):
    metrics_sum = {}
    metrics_total_samples = {}
    for client in client_results:
        num_samples = client['n_samples']
        metrics = client['metrics']
        for metric, value in metrics.items():
            if 'val' in metric:
                continue
            if metric not in metrics_sum:
                metrics_sum[metric] = 0.0
                metrics_total_samples[metric] = 0.0
            metrics_sum[metric] += value * num_samples
            metrics_total_samples[metric] += num_samples
    metrics_agg = {}
    for metric, total_value in metrics_sum.items():
        metrics_agg[metric] = total_value / metrics_total_samples[metric]
    return metrics_agg if metrics_agg else None

def get_train_history(client_results: List[Dict[str, Any]]):
    histories = {}
    for client in client_results:
        client_id = client['client_id']
        history = client['history'] if 'history' in client else None
        histories[client_id] = history
    return histories


def run_simulation(partitions: List,
                   config: Dict[str, Any],
                   hyperparameters: Dict[str, Any]):
    """Performs FL simulation."""

    # initialize variables
    n_clients = config['fl']['n_clients']
    n_rounds = config['fl']['n_rounds']
    save_history = config['fl']['save_history']

    # Ray initialisieren (wenn kein Cluster läuft, startet es einen lokalen)
    # include_dashboard=False unterdrückt das Starten des Web-Dashboards
    ray.init(ignore_reinit_error=True,
             include_dashboard=False,
             logging_level=logging.WARNING,  # Setzt den Level für Ray-Komponenten und oft auch für Worker
             log_to_driver=True         # Leitet Worker-Logs an den Driver (deine Konsole)
            )

    client_actors = []
    for i in range(n_clients):
        X_train, y_train, X_val, y_val = partitions[i]
        actor = ClientActor.remote(
            client_id=i,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            hyperparameters=hyperparameters
        )
        client_actors.append(actor)

    logging.info("Start FL simulation using Ray.")
    start_time = time.time()

    logging.info('[Server] Global model is initialized.')
    global_model = models.get_model(config=config,
                                    hyperparameters=hyperparameters)
    global_weights = global_model.get_weights()

    history = {}
    all_rounds_metrics_data = []
    for round_num in range(1, n_rounds+1):
        history[round_num] = {}
        logging.info(f"--- Round {round_num}/{n_rounds} ---")
        round_start_time = time.time()

        # Die .remote()-Aufrufe blockieren nicht, sie geben sofort ein ObjectRef zurück
        logging.info("[Server] Start train jobs on clients.")
        results_refs = []
        for actor in client_actors:
            # Übergebe die aktuellen globalen Gewichte an jeden Actor
            ref = actor.train.remote(global_weights)
            results_refs.append(ref)
        # Warte auf die Ergebnisse aller Clients und hole sie ab
        # ray.get() blockiert, bis alle Aufgaben in der Liste abgeschlossen sind
        client_results = ray.get(results_refs)
        logging.info(f"[Server] All clients trained. Start aggregation.")

        # Aggregate metrics and weights
        new_global_weights = aggregate_weights(client_results)
        train_metrics_agg = aggregate_metrics(client_results)

        # (optional:) save clients histories
        if save_history:
            history[round_num]['history'] = get_train_history(client_results)

        logging.info('[Server] Training metrics aggregated.')

        # Update global model
        if new_global_weights:
            global_model.set_weights(new_global_weights)
            global_weights = new_global_weights # Gewichte für die nächste Runde speichern
        else:
            logging.warning("[Server] There's nothing to aggregate.")

        logging.info(f'[Server] Start local evaluation.')
        # Local evaluation
        results_refs = []
        for actor in client_actors:
            # Übergebe die aktuellen globalen Gewichte an jeden Actor
            ref = actor.evaluate.remote(global_weights)
            results_refs.append(ref)
        client_results = ray.get(results_refs)
        #logging.info(f"[Server] Local evaluation terminated.")

        # aggregte evaluation metrics
        val_metrics_agg = aggregate_metrics(client_results)
        logging.info(f'[Server] Evaluation metrics aggregated.')

        round_data = {'round': round_num}
        if train_metrics_agg:
            for key, value in train_metrics_agg.items():
                if 'val' in key:
                    continue
                round_data[f'train_{key}'] = value
        if val_metrics_agg:
            for key, value in val_metrics_agg.items():
                round_data[f'eval_{key}'] = value
        all_rounds_metrics_data.append(round_data)

        round_end_time = time.time()
        logging.info(f"Round {round_num} terminated in {round_end_time - round_start_time:.2f} seconds.")

        # (Optional) Centralized evaluation

    end_time = time.time()
    logging.info(f"--- Simulation terminated in {end_time - start_time:.2f} seconds ---")
    ray.shutdown()
    metrics_df = pd.DataFrame(all_rounds_metrics_data)
    metrics_df = metrics_df.set_index('round') # Setze die Runde als Index
    history['metrics_aggregated'] = metrics_df
    logging.info("Aggregated Metrics DataFrame ---")
    logging.info(f"\n{metrics_df}")
    return history, global_model