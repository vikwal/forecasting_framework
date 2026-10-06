"""FedGradient (utils/fedgradient.py) and the park-id handling of the FL path.

Unit tests on synthetic inputs; the Ray end-to-end tests run on CPU. One optional test
reads two real parks_v1 parks (skipped when the data are not mounted).
"""

import copy
import os

import numpy as np
import pytest
import torch

from utils import fedgradient as fg
from utils import federated, preprocessing
from utils.models import TFT

LOOKBACK, HORIZON = 4, 3
QUANTILES = [0.5]


def _pinball(pred, tgt):
    from utils.tools import _pinball_loss
    return _pinball_loss(pred, tgt, QUANTILES)


def _tft(cards=(4,), dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return TFT(observed_dim=1, known_dim=2, static_dim=len(cards), hidden_dim=8, num_heads=2,
               lookback=LOOKBACK, horizon=HORIZON, dropout=dropout, static_embedding_dim=8,
               static_cardinalities=list(cards))


def _data(n, parks, seed=0):
    rng = np.random.default_rng(seed)
    X = {'observed': rng.normal(size=(n, LOOKBACK, 1)).astype(np.float32),
         'known': rng.normal(size=(n, LOOKBACK + HORIZON, 2)).astype(np.float32),
         'static': np.array([[float(parks[i % len(parks)])] for i in range(n)], dtype=np.float32)}
    y = rng.uniform(size=(n, HORIZON)).astype(np.float32)
    return X, y


def _client(model, X, y, bs=8, clip=None, cid='c'):
    return fg.GradientClient(model=model, X_train=X, y_train=y, batch_size=bs, criterion=_pinball,
                             device='cpu', is_tft=True, clipnorm=clip, median_idx=0, client_id=cid)


def _local_dispatch(clients, log=None):
    def dispatch(method, idx, arg):
        out = []
        for i in idx:
            c = clients[i]
            if method == 'prepare_round':
                out.append(c.prepare_round(*arg))
            elif method == 'step':
                batch = c.batches[c.cursor].tolist()
                r = c.step(arg)
                if log is not None:
                    log.append((i, batch))
                out.append(r)
            else:
                out.append(c.round_metrics())
        return out
    return dispatch


# ---------------------------------------------------------------------------
# Row-wise aggregation of the park-id embedding
# ---------------------------------------------------------------------------

EMB = 'static_embed.0.weight'


def _grads_with_rows(model, rows_value):
    """Gradient dict: zeros everywhere, given values in the embedding table."""
    g = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    g[EMB] = rows_value.clone()
    return g


def test_rowwise_disjoint_owners_each_row_gets_owner_gradient():
    model = _tft()
    n, d = model.static_embed[0].weight.shape
    ga = torch.full((n, d), 1.0)
    gb = torch.full((n, d), 5.0)        # foreign rows deliberately non-zero: must be ignored
    owners = {'A': {EMB: torch.tensor([True, True, False, False])},
              'B': {EMB: torch.tensor([False, False, True, True])}}
    res = [{'client_id': 'A', 'gradients': _grads_with_rows(model, ga), 'n_samples': 8},
           {'client_id': 'B', 'gradients': _grads_with_rows(model, gb), 'n_samples': 8}]
    grads, masks = fg.aggregate_gradients(res, owners)
    assert torch.equal(grads[EMB][:2], ga[:2]) and torch.equal(grads[EMB][2:], gb[2:])
    assert masks[EMB].all()
    # ordinary parameters: plain mean
    other = next(k for k in grads if k != EMB)
    assert torch.equal(grads[other], torch.zeros_like(grads[other]))


@pytest.mark.parametrize('opt', ['adam', 'adamw', 'sgd', 'sgd_momentum'])
def test_rows_without_contribution_stay_unchanged(opt):
    """A row whose owner does not participate keeps value and optimizer state, also with
    weight decay and momentum carried over from earlier steps."""
    model = _tft()
    hp = {'server_optimizer': opt, 'server_lr': 0.1, 'server_beta_1': 0.9, 'server_beta_2': 0.99,
          'server_eps': 1e-3, 'server_momentum': 0.9, 'server_weight_decay': 0.5}
    optim = fg.build_server_optimizer(model.parameters(), hp)
    owners = {'A': {EMB: torch.tensor([True, True, False, False])},
              'B': {EMB: torch.tensor([False, False, True, True])}}
    g = torch.randn(4, 8)
    both = [{'client_id': c, 'gradients': _grads_with_rows(model, g), 'n_samples': 8} for c in 'AB']
    fg.server_step(model, optim, *fg.aggregate_gradients(both, owners))   # builds momentum on all rows
    emb = model.static_embed[0].weight
    before = emb.detach().clone()
    state_before = {k: v.clone() for k, v in optim.state[emb].items() if torch.is_tensor(v) and v.dim()}
    only_a = [{'client_id': 'A', 'gradients': _grads_with_rows(model, g), 'n_samples': 8}]
    grads, masks = fg.aggregate_gradients(only_a, owners)
    assert masks[EMB].tolist() == [True, True, False, False]
    fg.server_step(model, optim, grads, masks)
    assert torch.equal(emb.detach()[2:], before[2:])                 # B's rows untouched
    assert not torch.equal(emb.detach()[:2], before[:2])             # A's rows moved
    for k, v in state_before.items():
        assert torch.equal(optim.state[emb][k][2:], v[2:])


def test_rows_owned_by_nobody_do_not_move_in_the_first_step():
    model = _tft()
    hp = {'server_optimizer': 'adamw', 'server_lr': 0.1, 'server_beta_1': 0.9, 'server_beta_2': 0.99,
          'server_eps': 1e-3, 'server_weight_decay': 0.5}
    optim = fg.build_server_optimizer(model.parameters(), hp)
    owners = {'A': {EMB: torch.tensor([True, False, False, False])}}
    emb = model.static_embed[0].weight
    before = emb.detach().clone()
    res = [{'client_id': 'A', 'gradients': _grads_with_rows(model, torch.ones(4, 8)), 'n_samples': 8}]
    fg.server_step(model, optim, *fg.aggregate_gradients(res, owners))
    assert torch.equal(emb.detach()[1:], before[1:])
    assert torch.count_nonzero(optim.state[emb]['exp_avg'][1:]) == 0


def test_fedavg_rowwise_takes_rows_from_owner():
    model = _tft()
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    wa = {k: v.clone() + 1.0 for k, v in ref.items()}
    wb = {k: v.clone() + 3.0 for k, v in ref.items()}
    owners = {'A': {EMB: torch.tensor([True, False, False, False])},
              'B': {EMB: torch.tensor([False, True, True, False])}}
    res = [{'client_id': 'A', 'weights': wa, 'n_samples': 10},
           {'client_id': 'B', 'weights': wb, 'n_samples': 30}]
    agg = federated.aggregate_weights(res, row_owners=owners, reference_weights=ref)
    assert torch.allclose(agg[EMB][0], wa[EMB][0]) and torch.allclose(agg[EMB][1:3], wb[EMB][1:3])
    assert torch.equal(agg[EMB][3], ref[EMB][3])                     # owned by nobody
    other = next(k for k in agg if k != EMB and agg[k].is_floating_point())
    assert torch.allclose(agg[other], ref[other] + 2.5)              # 0.25*1 + 0.75*3


def test_owned_rows_from_static_codes():
    X, _ = _data(10, parks=[1, 3])
    X['static'] += 1e-6                                              # float round-off
    rows = fg.owned_rows(X, {EMB: 0}, {EMB: 4})
    assert rows[EMB].tolist() == [False, True, False, True]
    with pytest.raises(ValueError):
        fg.owned_rows(X, {EMB: 0}, {EMB: 2})


# ---------------------------------------------------------------------------
# Park-id codes are global in FL
# ---------------------------------------------------------------------------

def _fl_config(categorical=True, val=None):
    cfg = {'data': {'files': ['P0', 'P1', 'P2', 'P3'], 'val_files': list(val or [])},
           'params': {'static_features': ['park_id'],
                      'static_categorical': ['park_id'] if categorical else []},
           'fl': {'clients': {'A': ['P0', 'P1'], 'B': ['P2', 'P3']}}}
    return cfg


def test_client_codes_come_from_the_global_list():
    cfg = _fl_config()
    codes = {}
    for cid, ids in cfg['fl']['clients'].items():
        ccfg = federated.client_data_config(cfg, ids)
        assert ccfg['data']['files'] == cfg['data']['files']          # never overwritten
        assert ccfg['data']['client_files'] == ids
        cats = preprocessing.static_categories(ccfg)['park_id']
        codes[cid] = [cats.index(p) for p in ids]
    assert codes == {'A': [0, 1], 'B': [2, 3]}                       # no collision


def test_holdout_codes_do_not_collide_with_clients():
    cfg = _fl_config(categorical=False, val=['H0', 'H1'])
    cfg['params']['static_categorical'] = ['park_id']
    cats = preprocessing.static_categories(cfg)['park_id']
    assert cats == ['P0', 'P1', 'P2', 'P3', 'H0', 'H1'] and len(set(cats)) == len(cats)


def test_check_client_parks():
    federated.check_client_parks(_fl_config())
    bad = _fl_config()
    bad['fl']['clients']['B'].append('PX')
    with pytest.raises(ValueError):
        federated.check_client_parks(bad)
    with pytest.raises(ValueError):                                  # park id + holdout
        federated.check_client_parks(_fl_config(val=['H0']))
    federated.check_client_parks(_fl_config(categorical=False, val=['H0']))


def _parks_v1_available():
    root = os.environ.get('DATA_ROOT')
    return bool(root) and os.path.isdir(os.path.join(root, 'synthetic/wind/parks_v1')) \
        and os.path.exists('data/parks_v1/wind_parameter.csv')


@pytest.mark.skipif(not _parks_v1_available(), reason='parks_v1 data not available')
def test_real_parks_get_global_codes():
    """Two parks of client N0 (positions 40, 41 in data.files) keep their global codes."""
    from utils import tools
    cfg = tools.load_config('configs/parks_v1/config_parks_v1_fl_fedgradient_parkid_smoke.yaml')
    cfg['model']['name'] = 'tft'
    cfg = tools.handle_freq(config=cfg)
    ids = cfg['fl']['clients']['N0']
    ccfg = federated.client_data_config(cfg, ids)
    dfs = preprocessing.get_data(data_dir=ccfg['data']['path'], config=ccfg, freq=ccfg['data']['freq'],
                                 features=preprocessing.get_features(config=ccfg), files_key='client_files')
    for pid in ids:
        df = dfs[f'synth_{pid}.csv']
        assert set(df['park_id'].unique()) == {cfg['data']['files'].index(pid)}
    assert sorted(cfg['data']['files'].index(p) for p in ids) == [2, 3]


# ---------------------------------------------------------------------------
# Epoch semantics
# ---------------------------------------------------------------------------

def test_epoch_batches_seeded_without_replacement():
    a = fg.epoch_batches(43, 8, seed=42001)
    assert len(a) == 5 and all(len(b) == 8 for b in a)
    flat = torch.cat(a).tolist()
    assert len(set(flat)) == 40
    assert [b.tolist() for b in a] == [b.tolist() for b in fg.epoch_batches(43, 8, seed=42001)]
    assert flat != torch.cat(fg.epoch_batches(43, 8, seed=42002)).tolist()
    assert fg.round_seed(42, 3) == 42003


def test_round_serves_every_batch_once_and_ends_when_all_clients_are_through():
    model = _tft()
    Xa, ya = _data(40, [0, 1], seed=1)                 # 5 batches
    Xb, yb = _data(27, [2, 3], seed=2)                 # 3 batches (drop_last)
    clients = [_client(_tft(seed=5), Xa, ya, cid='A'), _client(_tft(seed=6), Xb, yb, cid='B')]
    log = []
    optim = fg.build_server_optimizer(model.parameters(), {'server_optimizer': 'sgd', 'server_lr': 0.01})
    comm = fg.CommStats(['A', 'B'])
    metrics, steps = fg.run_round(_local_dispatch(clients, log), ['A', 'B'], model, optim,
                                  seed=42001, comm=comm)
    assert steps == 5
    served = {0: [], 1: []}
    for i, batch in log:
        served[i].extend(batch)
    assert sorted(served[0]) == list(range(40))
    assert len(served[1]) == 24 and len(set(served[1])) == 24
    assert [i for i, _ in log] == [0, 1, 0, 1, 0, 1, 0, 0]          # B drops out after 3 steps
    assert [m['n_samples'] for m in metrics] == [40, 24]
    assert comm.sync_steps == 5
    w = fg.tensor_bytes(fg.params_cpu(model))
    assert comm.download['A'] == 5 * w and comm.download['B'] == 3 * w
    assert comm.upload['B'] == 3 * w                                  # all params trainable


def test_exhausted_client_drops_out_of_the_mean():
    """Steps 4-5 use client A's gradient alone (not halved by the exhausted client)."""
    torch.manual_seed(0)
    Xa, ya = _data(40, [0, 1], seed=1)
    Xb, yb = _data(24, [2, 3], seed=2)
    model = _tft()
    ref = copy.deepcopy(model)
    clients = [_client(_tft(seed=5), Xa, ya, cid='A'), _client(_tft(seed=6), Xb, yb, cid='B')]
    hp = {'server_optimizer': 'sgd', 'server_lr': 0.05}
    fg.run_round(_local_dispatch(clients), ['A', 'B'], model,
                 fg.build_server_optimizer(model.parameters(), hp), seed=7)
    # reference: explicit loop
    opt = torch.optim.SGD(ref.parameters(), lr=0.05)
    ba, bb = fg.epoch_batches(40, 8, 7), fg.epoch_batches(24, 8, 7)
    ta, tb = fg.data_tensors(Xa, ya, 'cpu'), fg.data_tensors(Xb, yb, 'cpu')
    for k in range(5):
        parts = [(ta, ba[k])] + ([(tb, bb[k])] if k < 3 else [])
        grads = []
        for t, idx in parts:
            ref.zero_grad()
            b = [x[idx] for x in t]
            _pinball(ref(*b[:-1]), b[-1]).backward()
            grads.append({n: p.grad.clone() for n, p in ref.named_parameters()})
        for n, p in ref.named_parameters():
            p.grad = sum(g[n] for g in grads) / len(grads)
        opt.step()
    for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6), n


# ---------------------------------------------------------------------------
# Equivalence with centralized training
# ---------------------------------------------------------------------------

def test_one_client_equals_centralized_training():
    """1 client, server Adam = CL Adam (tools.training_pipeline), same seed and batch
    order, clipping as model.tft.clipnorm: identical weights after two epochs."""
    X, y = _data(64, [0, 1, 2, 3], seed=3)
    lr, wd, clip, bs = 1e-3, 1e-4, 0.5, 8
    model = _tft(seed=11)
    cl_model = copy.deepcopy(model)
    hp = {'server_optimizer': 'adam', 'server_lr': lr, 'server_beta_1': 0.9, 'server_beta_2': 0.999,
          'server_eps': 1e-8, 'server_weight_decay': wd}
    optim = fg.build_server_optimizer(model.parameters(), hp)
    client = _client(_tft(seed=99), X, y, bs=bs, clip=clip, cid='only')
    for rnd in (1, 2):
        fg.run_round(_local_dispatch([client]), ['only'], model, optim, seed=fg.round_seed(42, rnd))

    cl_opt = torch.optim.Adam(cl_model.parameters(), lr=lr, weight_decay=wd)   # as tools.training_pipeline
    tensors = fg.data_tensors(X, y, 'cpu')
    for rnd in (1, 2):
        for idx in fg.epoch_batches(64, bs, fg.round_seed(42, rnd)):
            b = [t[idx] for t in tensors]
            cl_model.train()
            cl_opt.zero_grad()
            _pinball(cl_model(*b[:-1]), b[-1]).backward()
            torch.nn.utils.clip_grad_norm_(cl_model.parameters(), clip)
            cl_opt.step()
    for (n, p), (_, q) in zip(model.named_parameters(), cl_model.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), n


# ---------------------------------------------------------------------------
# Server optimizer from the config
# ---------------------------------------------------------------------------

def _fg_cfg(**over):
    cfg = {'server_optimizer': 'adam', 'server_lr': 0.01, 'beta_1': 0.9, 'beta_2': 0.99,
           'eps': 0.001, 'momentum': 0.8, 'weight_decay': 0.0, 'clipnorm': 1.0}
    cfg.update(over)
    return {'fl': {'strategy': 'fedgradient', 'fedgradient': cfg}, 'model': {'tft': {'clipnorm': 7.0}}}


@pytest.mark.parametrize('name,cls', [('adam', torch.optim.Adam), ('adamw', torch.optim.AdamW),
                                      ('sgd', torch.optim.SGD), ('sgd_momentum', torch.optim.SGD)])
def test_server_optimizer_from_config(name, cls):
    hp = fg.fedgradient_hyperparameters(_fg_cfg(server_optimizer=name, server_lr=0.003, weight_decay=0.01))
    opt = fg.build_server_optimizer(_tft().parameters(), hp)
    assert type(opt) is cls
    group = opt.param_groups[0]
    assert group['lr'] == 0.003 and group['weight_decay'] == 0.01
    if name in ('adam', 'adamw'):
        assert group['betas'] == (0.9, 0.99) and group['eps'] == 0.001
    elif name == 'sgd_momentum':
        assert group['momentum'] == 0.8
    else:
        assert group['momentum'] == 0


def test_fedgradient_config_validation_and_clip_fallback():
    with pytest.raises(ValueError):
        fg.fedgradient_hyperparameters(_fg_cfg(server_optimizer='rmsprop'))
    with pytest.raises(ValueError):
        fg.fedgradient_hyperparameters(_fg_cfg(beta_2=None))
    cfg = _fg_cfg()
    del cfg['fl']['fedgradient']['clipnorm']
    assert fg.fedgradient_hyperparameters(cfg)['client_clipnorm'] == 7.0       # model.tft.clipnorm
    assert fg.fedgradient_hyperparameters(_fg_cfg(clipnorm=None))['client_clipnorm'] is None


def _parks_fl_config():
    from utils import tools
    os.environ.setdefault('DATA_ROOT', '/nonexistent')
    cfg = tools.load_config('configs/parks_v1/config_parks_v1_fl_fedgradient.yaml')
    cfg['model']['name'] = 'tft'
    cfg['model']['fl'] = True
    return cfg


def test_hyperparameters_carry_server_settings():
    """Regression: server_lr must come from fl.fedgradient, not the old 1.0 default."""
    from utils import hpo
    hp = hpo.get_hyperparameters(config=_parks_fl_config())
    assert hp['strategy'] == 'fedgradient'
    assert hp['server_lr'] == 0.01 and hp['server_optimizer'] == 'adam'
    assert (hp['server_beta_1'], hp['server_beta_2'], hp['server_eps']) == (0.9, 0.99, 0.001)
    assert hp['client_clipnorm'] == 1.0 and hp['lr'] == 0.0005


def test_hpo_trial_samples_fedgradient_dimensions():
    import optuna
    from utils import hpo
    cfg = _parks_fl_config()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(5):
        hp = hpo.get_hyperparameters(config=cfg, hpo=True, trial=study.ask())
        assert hp['server_optimizer'] in ('adam', 'sgd_momentum')
        assert 1e-4 <= hp['server_lr'] <= 0.1 and hp['server_lr'] != 0.01
        if hp['server_optimizer'] == 'sgd_momentum':
            assert hp['server_momentum'] == 0.9
        else:
            assert hp['server_beta_2'] == 0.99


# ---------------------------------------------------------------------------
# End to end through run_simulation (Ray, CPU)
# ---------------------------------------------------------------------------

def _sim_config(strategy, tmp_path=None, n_rounds=2, cards=(4,)):
    cfg = {'model': {'name': 'tft', 'lookback': LOOKBACK, 'horizon': HORIZON, 'shuffle': True,
                     'feature_dim': {'observed_dim': 1, 'known_dim': 2, 'static_dim': 1},
                     'tft': {'quantiles': QUANTILES, 'rnn_type': 'lstm', 'clipnorm': 7.0}},
           'data': {'files': ['P0', 'P1', 'P2', 'P3']},
           'params': {'random_seed': 42, 'static_features': ['park_id'],
                      'static_categorical': ['park_id'] if cards[0] else []},
           'fl': {'strategy': strategy, 'n_rounds': n_rounds, 'n_local_epochs': 1,
                  'max_concurrent_actors': 2, 'global_early_stopping': {'enabled': True, 'patience': 5},
                  'fedopt': {'server_lr': 0.01, 'beta_1': 0.9, 'beta_2': 0.99, 'tau': 0.001},
                  'fedgradient': {'server_optimizer': 'adam', 'server_lr': 0.01, 'beta_1': 0.9,
                                  'beta_2': 0.99, 'eps': 0.001, 'clipnorm': 1.0}}}
    if tmp_path is not None:
        cfg['fl']['checkpoint'] = {'enabled': True, 'dir': str(tmp_path)}
    return cfg


def _sim_hp(strategy):
    hp = {'batch_size': 8, 'lr': 1e-3, 'hidden_dim': 8, 'n_heads': 2, 'dropout': 0.0,
          'static_embedding_dim': 8, 'n_rounds': None}
    if strategy == 'fedadam':
        hp.update(server_lr=0.01, beta_1=0.9, beta_2=0.99, tau=0.001)
    return hp


def _partitions():
    Xa, ya = _data(32, [0, 1], seed=1)
    Xb, yb = _data(24, [2, 3], seed=2)
    Va, va = _data(8, [0, 1], seed=3)
    Vb, vb = _data(8, [2, 3], seed=4)
    return {'A': (Xa, ya, Va, va), 'B': (Xb, yb, Vb, vb)}


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    yield


def _run(strategy, n_rounds=2, tmp_path=None, resume=None):
    cfg = _sim_config(strategy, tmp_path, n_rounds)
    if resume:
        cfg['fl']['resume'] = resume
    hp = _sim_hp(strategy)
    hp['n_rounds'] = n_rounds
    return federated.run_simulation(_partitions(), cfg, hp)


@pytest.mark.parametrize('strategy', ['fedavg', 'fedadam', 'fedgradient'])
def test_run_simulation_smoke(cpu_only, strategy):
    history, weights = _run(strategy)
    df = history['metrics_aggregated']
    assert list(df.index) == [1, 2] and np.isfinite(df['val_rmse']).all()
    assert history['last_round'] == 2
    assert set(weights) == {'A', 'B'}
    assert history['comm_stats']['sync_steps'] == (8 if strategy == 'fedgradient' else 2)  # 4 batches/round
    assert history['comm_stats']['total_upload_bytes'] > 0


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        federated.run_simulation(_partitions(), _sim_config('fedsgd'), _sim_hp('fedsgd'))


def test_fedgradient_resume_equals_uninterrupted_run(cpu_only, tmp_path):
    full, w_full = _run('fedgradient', n_rounds=3)
    _run('fedgradient', n_rounds=2, tmp_path=tmp_path)
    ck = fg.load_checkpoint(os.path.join(tmp_path, 'last.pt'))
    assert ck['round'] == 2 and ck['server_optimizer']['state']
    resumed, w_res = _run('fedgradient', n_rounds=3, resume=os.path.join(tmp_path, 'last.pt'))
    assert list(resumed['metrics_aggregated'].index) == [1, 2, 3] and resumed['last_round'] == 3
    assert resumed['comm_stats']['sync_steps'] == full['comm_stats']['sync_steps']
    best_full = full['best_round']
    assert resumed['best_round'] == best_full
    for k in w_full['A']:
        assert torch.allclose(w_full['A'][k], w_res['A'][k], atol=1e-6), k


def test_last_round_is_the_stopping_round(cpu_only):
    cfg = _sim_config('fedgradient', n_rounds=6)
    cfg['fl']['global_early_stopping'] = {'enabled': True, 'patience': 1, 'min_delta': 10.0}
    hp = _sim_hp('fedgradient')
    hp['n_rounds'] = 6
    history, _ = federated.run_simulation(_partitions(), cfg, hp)
    # round 1 improves on inf, round 2 cannot beat it by min_delta -> stop after round 2
    assert history['early_stopped'] and history['last_round'] == 2 and history['best_round'] == 1
    assert list(history['metrics_aggregated'].index) == [1, 2]
