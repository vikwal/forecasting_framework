import ray
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Any
from sklearn.model_selection import TimeSeriesSplit

from . import tools, models

class ServerOptimizer:
    """
    Verwaltet den Zustand und die Update-Logik für serverseitige Optimierer
    wie FedAvgM und FedAdam.
    """
    def __init__(self,
                 strategy: str,
                 hyperparameters: Dict[str, Any],
                 initial_weights: List[np.ndarray]):
        self.strategy = strategy.lower()
        self.hyperparameters = hyperparameters
        self.step_count = 0

        # Hyperparameter für Server-Optimierer holen (mit Standardwerten)
        self.server_lr = hyperparameters.get('server_lr', 1.0)      # Server Lernrate (eta_s)
        self.beta_1 = hyperparameters.get('beta_1', 0.9)            # Momentum für FedAvgM, Beta1 für FedAdam
        self.beta_2 = hyperparameters.get('beta_2', 0.999)          # Beta2 für FedAdam
        self.tau = hyperparameters.get('tau', 1e-8)                 # Epsilon/Tau für FedAdam (numerische Stabilität)

        # Initialisiere Zustandsvariablen (als Liste von Numpy-Arrays, wie die Gewichte)
        self.server_state_v = [np.zeros_like(w) for w in initial_weights] # Momentum für FedAvgM / v für FedAdam
        if self.strategy in ['fedadam', 'fedyogi']: # FedAdam/Yogi brauchen einen zweiten Moment-Schätzer
            self.server_state_m = [np.zeros_like(w) for w in initial_weights] # m für FedAdam/Yogi

        logging.info(f"[ServerOptimizer] Initialized for strategy '{self.strategy}' with server_lr={self.server_lr}, "
                     f"beta_1={self.beta_1}, beta_2={self.beta_2}, tau={self.tau}")

    def step(self,
             old_global_weights: List[np.ndarray],
             aggregated_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Führt einen Optimierungsschritt auf dem Server aus.

        Args:
            old_global_weights (w_t): Die globalen Gewichte VOR dem Update.
            aggregated_weights (w_{t+1}^{avg}): Die durch FedAvg aggregierten Gewichte.

        Returns:
            List[np.ndarray]: Die finalen globalen Gewichte (w_{t+1}) nach dem Server-Optimierungsschritt.
        """
        self.step_count += 1
        # Berechne das durchschnittliche Update-Delta (w_{t+1}^{avg} - w_t)
        delta_t = [agg - old for agg, old in zip(aggregated_weights, old_global_weights)]

        if self.strategy == 'fedavgm':
            # FedAvgM Update Rule:
            # v_{t+1} = beta_1 * v_t + delta_t
            # w_{t+1} = w_t + server_lr * v_{t+1}
            # Update server momentum (v) -> In-place Update der Liste
            self.server_state_v[:] = [self.beta_1 * v + dt for v, dt in zip(self.server_state_v, delta_t)]
            # Calculate new global weights
            final_weights = [old + self.server_lr * v for old, v in zip(old_global_weights, self.server_state_v)]
            logging.debug(f"[ServerOptimizer FedAvgM] Applied momentum update. Step: {self.step_count}")

        elif self.strategy == 'fedadam':
            # FedAdam Update Rule:
            # m_{t+1} = beta_1 * m_t + (1 - beta_1) * delta_t
            # v_{t+1} = beta_2 * v_t + (1 - beta_2) * delta_t^2
            # Optional: Bias correction (hier implementiert)
            # m_hat = m_{t+1} / (1 - beta_1^t)
            # v_hat = v_{t+1} / (1 - beta_2^t)
            # w_{t+1} = w_t + server_lr * m_hat / (sqrt(v_hat) + tau)
            # Update first moment estimate (m) -> In-place
            self.server_state_m[:] = [self.beta_1 * m + (1 - self.beta_1) * dt
                                      for m, dt in zip(self.server_state_m, delta_t)]
            # Update second moment estimate (v) -> In-place
            self.server_state_v[:] = [self.beta_2 * v + (1 - self.beta_2) * np.square(dt)
                                      for v, dt in zip(self.server_state_v, delta_t)]
            beta_1_power = self.beta_1 ** self.step_count
            beta_2_power = self.beta_2 ** self.step_count
            m_hat = [m / (1 - beta_1_power) for m in self.server_state_m]
            v_hat = [v / (1 - beta_2_power) for v in self.server_state_v]

            final_weights = [old + self.server_lr * m_h / (np.sqrt(v_h) + self.tau)
                             for old, m_h, v_h in zip(old_global_weights, m_hat, v_hat)]
            logging.debug(f"[ServerOptimizer FedAdam] Applied Adam update. Step: {self.step_count}")

        elif self.strategy == 'fedyogi':
             # FedYogi ist ähnlich zu FedAdam, aber mit anderem Update für v
             # v_{t+1} = v_t - (1 - beta_2) * delta_t^2 * sign(v_t - delta_t^2)
             # (Implementierung hier ausgelassen, da nicht primär gefordert)
             logging.warning(f"FedYogi strategy not fully implemented yet.")
             final_weights = aggregated_weights # Fallback zu FedAvg

        elif self.strategy == 'fedadagrad':
             # FedAdagrad braucht auch einen state (Summe der quadrierten Deltas)
             # v_{t+1} = v_t + delta_t^2
             # w_{t+1} = w_t + server_lr * delta_t / (sqrt(v_{t+1}) + tau)
             # (Implementierung hier ausgelassen)
             logging.warning(f"FedAdagrad strategy not fully implemented yet.")
             final_weights = aggregated_weights # Fallback zu FedAvg
        else:
             logging.error(f"Unknown server optimization strategy: {self.strategy}")
             final_weights = aggregated_weights
        return [np.array(w) for w in final_weights]

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
        self.config = config
        self.hyperparameters = hyperparameters
        # Wichtig: Das Modell muss innerhalb des Actors erstellt werden
        tools.initialize_gpu()
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
            verbose=0#self.config['model']['verbose'],
        )
        if verbose:
            logging.info(f"[ClientActor {self.client_id}] Terminated fitting.")
        metrics = {}
        for key, value in history.history.items():
            metrics[key] = value[-1] # get metric in last round
        results = {'client_id': self.client_id,
                   'weights': self.model.get_weights(),
                   'n_samples': len(self.y_train),
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
                verbose=0,#self.config['model']['verbose'],
                return_dict=True
            )
        else:
            logging.warning(f"[ClientActor {self.client_id}] No validation data available for evaluation.")
            metrics = None
        logging.info(f"[ClientActor {self.client_id}] Terminated local evaluation.")
        results = {'client_id': self.client_id,
                   'n_samples': len(self.y_val),
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

def get_kfolds_partitions(n_splits, partitions):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    raw_partitions = [] # Speichert Folds pro ursprünglicher Partition: [[fold1_p1, fold2_p1,...], [fold1_p2, ...]]
    for i, part in enumerate(partitions):
        X_train_orig = part[0] # Kann np.ndarray oder dict sein
        y_train = part[1]      # Sollte immer np.ndarray sein
        folds = [] # Speichert die Folds für DIESE eine Partition part
        data_for_splitting = y_train
        for fold_idx, (train_index, val_index) in enumerate(tscv.split(data_for_splitting)):
            # 1. Teile y_train (ist immer ein Array)
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            if isinstance(X_train_orig, np.ndarray):
                X_train_fold = X_train_orig[train_index]
                X_val_fold = X_train_orig[val_index]
            elif isinstance(X_train_orig, dict):
                # Case for TFT
                X_train_fold = {}
                X_val_fold = {}
                for key, value_array in X_train_orig.items():
                    # Prüfe, ob der Key 'static_input' ist
                    if key == 'static_input':
                        # Statische Features werden nicht gesplittet, sondern übernommen
                        X_train_fold[key] = value_array
                        X_val_fold[key] = value_array
                        logging.debug(f"Fold {fold_idx}, Partition {i}: Passing through static_input.")
                    else:
                        X_train_fold[key] = value_array[train_index]
                        X_val_fold[key] = value_array[val_index]
            folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        raw_partitions.append(folds)

    kfolds_partitions = []
    for split_idx in range(n_splits): # Iteriere über die Fold-Indizes (0 bis K-1)
        partition_for_this_fold = []
        for client_folds in raw_partitions: # Iteriere über die Ergebnisse jeder ursprünglichen Partition
            # Stelle sicher, dass diese Partition Folds hat und auch diesen spezifischen Fold
            partition_for_this_fold.append(client_folds[split_idx])
        kfolds_partitions.append(partition_for_this_fold)
    return kfolds_partitions


def run_simulation(partitions: List,
                   config: Dict[str, Any],
                   hyperparameters: Dict[str, Any]):
    """Performs FL simulation."""

    # initialize variables
    n_clients = config['fl']['n_clients']
    n_rounds = hyperparameters['n_rounds']
    save_history = config['fl']['save_history']
    strategy = config['fl']['strategy'].lower()
    total_num_gpus = len(tf.config.list_physical_devices('GPU'))
    gpu_per_actor = min(1, total_num_gpus/n_clients)

    # Ray initialisieren (wenn kein Cluster läuft, startet es einen lokalen)
    # include_dashboard=False unterdrückt das Starten des Web-Dashboards
    ray.init(num_gpus=total_num_gpus,
             ignore_reinit_error=True,
             include_dashboard=False,
             logging_level=logging.WARNING,  # Setzt den Level für Ray-Komponenten und oft auch für Worker
             log_to_driver=True         # Leitet Worker-Logs an den Driver (deine Konsole)
            )

    client_actors = []
    for i in range(n_clients):
        X_train, y_train, X_val, y_val = partitions[i]
        actor = ClientActor.options(num_gpus=gpu_per_actor).remote(
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

    server_optimizer = None
    if strategy != 'fedavg':
        server_optimizer = ServerOptimizer(strategy, hyperparameters, global_weights)

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
        aggregated_weights = aggregate_weights(client_results)
        train_metrics_agg = aggregate_metrics(client_results)

        # --- Server-seitigen Optimierungsschritt anwenden (falls nicht FedAvg) ---
        if server_optimizer:
            new_global_weights = server_optimizer.step(global_weights, aggregated_weights)
        else:
            # Bei FedAvg sind die aggregierten Gewichte die finalen Gewichte
            new_global_weights = aggregated_weights

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