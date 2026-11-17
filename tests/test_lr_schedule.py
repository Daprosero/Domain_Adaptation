"""The annealed phase-2 learning rate (T-GARNet / DANN schedule) applied through an optimizer hook."""
import pytest
import torch

from MIL_CREDA_EEG import lr_schedule as S


def _steps(opt, n):
    seen = []
    for _ in range(n):
        opt.zero_grad()
        opt.param_groups[0]["params"][0].sum().backward()
        opt.step()
        seen.append(opt.param_groups[0]["lr"])
    return seen


def test_annealed_follows_the_schedule_on_the_lr2_optimizer():
    p = torch.nn.Parameter(torch.ones(3))
    with S.annealed(1e-3, [10]):
        opt = torch.optim.Adam([p], lr=1e-3)
        seen = _steps(opt, 12)
    expected = [1e-3 * (1 + 10 * min(1.0, n / 10)) ** -0.75 for n in range(12)]
    assert seen == pytest.approx(expected)


def test_optimizers_at_another_lr_are_left_alone():
    p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    with S.annealed(1e-3, [5]):
        main, dual = torch.optim.SGD([p], lr=1e-3), torch.optim.SGD([q], lr=0.5)
        _steps(main, 3)
        assert _steps(dual, 3) == [0.5, 0.5, 0.5]


def test_each_lr2_optimizer_takes_its_own_total_in_creation_order():
    p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    with S.annealed(1e-2, [4, 8]):
        a, b = torch.optim.Adam([p], lr=1e-2), torch.optim.Adam([q], lr=1e-2)
        la, lb = _steps(a, 4), _steps(b, 4)
    assert la[-1] == pytest.approx(1e-2 * (1 + 10 * 3 / 4) ** -0.75)
    assert lb[-1] == pytest.approx(1e-2 * (1 + 10 * 3 / 8) ** -0.75)


def test_count_steps_and_no_effect_outside_the_context():
    p = torch.nn.Parameter(torch.ones(2))
    with S.count_steps(1e-3) as counts:
        _steps(torch.optim.Adam([p], lr=1e-3), 7)
        _steps(torch.optim.Adam([p], lr=0.1), 2)
    assert counts == [7]
    assert _steps(torch.optim.Adam([p], lr=1e-3), 3) == [1e-3] * 3


def test_annealed_refuses_an_unscheduled_run():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(S.NotScheduled):
        with S.annealed(1e-3, [5]):
            _steps(torch.optim.Adam([p], lr=0.1), 2)
