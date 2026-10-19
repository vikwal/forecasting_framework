"""Component-selective freezing for TFT fine-tuning.

Variants are defined as sets of *top-level module names* of utils/models.py
TFT (exact match on the first state_dict path segment — no substring matching,
so e.g. 'lstm_gate' can never be caught by 'output_gate' patterns).
"""

STATIC_PATH = {
    'static_embed', 'static_variable_selection', 'static_context_grn',
    'static_enrichment_grn', 'static_state_h_grn', 'static_state_c_grn',
}

HEAD = {'output_layer', 'output_gate', 'output_ln', 'positionwise_grn'}

TEMPORAL_VSN = {'vsn_past', 'vsn_future', 'observed_embed', 'known_embed'}

# variant name -> set of trainable top-level modules (None = all trainable)
VARIANTS = {
    'full': None,
    'head': HEAD,
    'static': STATIC_PATH,
    'static_vsn': STATIC_PATH | TEMPORAL_VSN,
}


def apply_freeze(model, variant: str) -> dict:
    """Freeze all parameters except those of the variant's modules.

    Returns a summary dict with the trainable module names and counts.
    """
    allow = VARIANTS[variant]
    trainable_modules = set()
    n_trainable, n_total = 0, 0
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        train_it = allow is None or top in allow
        p.requires_grad_(train_it)
        n_total += p.numel()
        if train_it:
            n_trainable += p.numel()
            trainable_modules.add(top)
    if allow is not None:
        # Every requested module must actually exist on this model
        # (static_state_c_grn is absent for rnn_type='gru' — tolerated).
        missing = {m for m in allow if m != 'static_state_c_grn'} - trainable_modules
        assert not missing, f'freeze variant {variant}: modules not found: {missing}'
    return {
        'variant': variant,
        'trainable_modules': sorted(trainable_modules),
        'trainable_params': n_trainable,
        'total_params': n_total,
    }


def self_test():
    """Unit test: two optimizer steps per variant must change exactly the
    allowed modules and leave every other tensor bit-identical."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from utils.models import TFT

    torch.manual_seed(0)
    model_kwargs = dict(observed_dim=1, known_dim=6, static_dim=7,
                        hidden_dim=16, num_heads=2, lookback=48, horizon=48,
                        dropout=0.1, num_lstm_layers=1, static_embedding_dim=32)
    obs = torch.randn(8, 48, 1)
    known = torch.randn(8, 96, 6)
    static = torch.randn(8, 7)
    target = torch.randn(8, 48)

    param_counts = {}
    for variant in VARIANTS:
        model = TFT(**model_kwargs)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        info = apply_freeze(model, variant)
        param_counts[variant] = info['trainable_params']
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(2):
            loss = torch.nn.functional.mse_loss(model(obs, known, static), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        allow = VARIANTS[variant]
        changed_modules = set()
        for k, v in model.state_dict().items():
            if not torch.equal(before[k], v):
                changed_modules.add(k.split('.')[0])
        if allow is None:
            assert len(changed_modules) > 10, 'full variant barely trained'
        else:
            illegal = changed_modules - allow
            assert not illegal, f'{variant}: frozen modules changed: {illegal}'
            assert changed_modules, f'{variant}: nothing changed'
        print(f'  {variant:<11} trainable={info["trainable_params"]:>7,} '
              f'changed_modules={sorted(changed_modules)}')

    assert param_counts['head'] < param_counts['static'] \
        < param_counts['static_vsn'] < param_counts['full'], \
        f'parameter counts not strictly ordered: {param_counts}'
    print('freezing self-test passed.')


if __name__ == '__main__':
    self_test()
