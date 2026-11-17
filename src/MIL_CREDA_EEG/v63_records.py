"""Plan v63: what each refit record keeps so Secs. 5.3, 6.1, 6.2 and 6.4 read it without
retraining.

Per target test bag of a record:

- `windows`: the pointer of every window the model read, `(subject, condition, index)` with
  `index` the position in the bag's stored windows (the step of ceil(n/60) of Sec. 7.3, so a
  PLV analysis reloads the same window from the processed `.npz`), its start time in
  seconds, and its markers: bad contact on any channel the transfer uses (EMOTIV), the
  event mark (the high-density task), the integrity mark and the rail mark;
- `attention`: the aggregation weight of every window (the selector of MIL-CREDA and of the
  common wrapper, ABMIL's for AdaptMIL), in the same order;
- for MIL-CREDA and its K3 variants, `correspondence`: the local correspondence pi
  (Eq. 35 as run: g without gradient, tau_loc the row spread) of the bag over the source
  training bags of the outer fold, whose subjects and labels are stored once per record.

Nothing here trains or chooses; it reads the trained model once, without gradient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG import mil_creda_model as M
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["5.3", "6.1", "6.2", "6.4", "7.3"],
    "experiments": ["C-dispositivo", "K1-atencion", "K2-emparejamiento", "K4-conectividad"],
    "invariants": [
        "the_window_pointers_are_the_windows_the_model_read",
        "the_attention_weights_of_a_bag_sum_to_one",
        "the_correspondence_of_a_target_bag_is_a_distribution_over_the_source_bags",
    ],
}


def selected_indices(n: int) -> list[int]:
    """The stored-window positions `v63_setup.prepare` keeps."""
    return list(range(0, int(n), V.window_step(n)))[:V.WINDOWS]


def window_info(raw, channels: Sequence[str] | None) -> dict:
    """Pointers and markers of the windows of one raw bag the model read."""
    marks = raw.marks()
    n = len(marks["window_index"])
    idx = selected_indices(n)
    names = list(raw.channels)
    cols = [names.index(c) for c in channels] if channels else list(range(len(names)))
    flag = lambda key: ([bool(v) for v in np.asarray(marks[key])[idx]]
                        if key in marks else None)
    contact = (marks["contact_bad"][idx][:, cols].any(1).tolist()
               if "contact_bad" in marks else None)
    return {"subject": raw.subject, "condition": raw.condition, "index": idx,
            "storedIndex": [int(v) for v in np.asarray(marks["window_index"])[idx]],
            "startSeconds": [float(v) for v in np.asarray(marks["window_start_s"])[idx]],
            "contactBad": contact, "event": flag("event_window"),
            "integrity": flag("integrity_mark"), "rail": flag("rail_window")}


@torch.no_grad()
def attention(model, bags: Sequence[two_phase.BagTensor], domain: str, device) -> list:
    """Per bag, the aggregation weight of each window."""
    model.eval()
    out = []
    for b in bags:
        H = model.embed([b.windows.to(device)], domain, two_phase.EVAL_CHUNK)[0]
        if hasattr(model, "aggregator"):                    # AdaptMIL's native ABMIL
            w = model.aggregator(model.aggregator.project(H))[1]
        else:
            w = model.bag_pairs([H])[0][1]
        out.append([float(v) for v in w.detach().cpu()])
    return out


@torch.no_grad()
def correspondence(model, source: Sequence[two_phase.BagTensor],
                   target: Sequence[two_phase.BagTensor], device) -> dict:
    """Eq. 35 as run for each target bag over `source` (MIL-CREDA-shaped models)."""
    model.eval()
    dev = torch.device(device)
    pairs_s = model.bag_pairs(model.embed([b.windows.to(dev) for b in source], "source",
                                          two_phase.EVAL_CHUNK))
    pairs_t = model.bag_pairs(model.embed([b.windows.to(dev) for b in target], "target",
                                          two_phase.EVAL_CHUNK))
    K_st = M.bag_kernel_matrix(pairs_s, pairs_t, model.sigma)
    G_t = model.classify(model.represent(pairs_t))
    labels = torch.tensor([b.label for b in source], device=K_st.device)
    pi = []
    for j in range(len(target)):
        cross = K_st[:, j]
        pi.append([float(v) for v in V._corr(cross, labels, G_t[j].detach(), V.row_tau(cross))])
    return {"correspondence": pi, "correspondenceSourceSubjects": [b.subject for b in source],
            "correspondenceSourceLabels": [int(b.label) for b in source]}


def target_extras(model, test: Sequence[two_phase.BagTensor], bags, channels, domain: str,
                  device, source_train: Sequence[two_phase.BagTensor] | None = None) -> dict:
    out = {"windows": [window_info(bags.get(b.subject, b.condition), channels) for b in test],
           "attention": attention(model, test, domain, device)}
    for b, w, a in zip(test, out["windows"], out["attention"]):
        if not (len(w["index"]) == len(a) == int(b.windows.shape[0])):
            raise V.V63Error(f"{b.subject}: the window pointers do not match the windows read")
    if source_train is not None:
        out.update(correspondence(model, source_train, test, device))
    return out
