"""The contamination axis: what it replaces, what it leaves alone, and the zero.

Bags are pure and no instance carries a label of its own, so contamination is
not a flip. It replaces part of a *training* bag with images of other classes
and leaves the bag's label untouched: what is corrupted is the evidence a bag
offers for its own label, never the answer it is measured against.

Every test here is built so it could come out red. The one that matters most is
the zero -- at rate 0 the material has to be what it was before this axis
existed, image for image, or no clean number stays comparable to itself.
"""

import re

import numpy as np
import pytest

from MIL_CREDA_Benchmark import bags, config


@pytest.fixture
def labels():
    """A domain with enough of every class to fund the bags and the reserve."""
    return np.repeat(np.arange(config.CLASSES), 2000)


def _material(rate, seed=0, code="M"):
    """The draw and the contamination, without decoding a single image."""
    rng = np.random.default_rng(seed * 1000 + ord(code))
    chosen, spare = bags._select(np.repeat(np.arange(config.CLASSES), 2000),
                                 rng, bags._reserve_size(rate))
    flat, members, bag_labels = [], [], []
    train, valid, evaluation = [], [], []
    train_share, valid_share = bags._split_counts(rng)
    for class_id in range(config.CLASSES):
        offset = len(flat)
        flat.extend(int(i) for i in chosen[class_id])
        for k in range(config.BAGS_PER_CLASS):
            start = offset + k * config.INSTANCES_PER_BAG
            members.append(list(range(start, start + config.INSTANCES_PER_BAG)))
            bag_labels.append(class_id)
            position = len(members) - 1
            if k < train_share[class_id]:
                train.append(position)
            elif k < train_share[class_id] + valid_share[class_id]:
                valid.append(position)
            else:
                evaluation.append(position)
    before = list(flat)
    record = bags._contaminate(flat, members, bag_labels, train, spare,
                               rate, seed, code)
    return {"before": before, "after": flat, "members": members,
            "labels": bag_labels, "train": train, "valid": valid,
            "eval": evaluation, "record": record}


class TestTheZero:
    """Rate 0 has to leave the draw exactly where it found it."""

    def test_a_clean_draw_takes_no_reserve_at_all(self):
        """A reserve at rate 0 would move every seed's images by its own width."""
        assert bags._reserve_size(0.0) == 0

    def test_the_clean_draw_is_the_draw_that_predates_the_axis(self, labels):
        """`_select` with no reserve draws what the old two-line body drew.

        Written against the old behaviour rather than against the new code, so it
        fails if the permutation is ever consumed differently -- which is the one
        change that would silently invalidate every clean number on record.
        """
        needed = config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG
        expected = {}
        rng = np.random.default_rng(11)
        for class_id in range(config.CLASSES):
            pool = np.flatnonzero(labels == class_id)
            expected[class_id] = rng.permutation(pool)[:needed]

        chosen, spare = bags._select(labels, np.random.default_rng(11), 0)
        for class_id in range(config.CLASSES):
            assert np.array_equal(chosen[class_id], expected[class_id])
            assert spare[class_id].size == 0

    def test_contaminating_at_zero_changes_no_image(self):
        drawn = _material(0.0)
        assert drawn["after"] == drawn["before"]
        assert drawn["record"]["instancesPerBag"] == 0
        assert drawn["record"]["bags"] == []


class TestWhatItReplaces:
    """The replacement itself, at a rate that has to do something."""

    RATE = 0.4

    def test_every_training_bag_loses_exactly_the_declared_count(self):
        drawn = _material(self.RATE)
        expected = config.noise_instances(self.RATE)
        assert expected > 0, "a rate that replaces nothing proves nothing below"
        for position in drawn["train"]:
            slots = drawn["members"][position]
            moved = sum(1 for s in slots
                        if drawn["after"][s] != drawn["before"][s])
            assert moved == expected

    def test_the_other_two_roles_are_untouched(self):
        """`valid` reads the search's criterion and `eval` is the answer key."""
        drawn = _material(self.RATE)
        for role in ("valid", "eval"):
            for position in drawn[role]:
                for slot in drawn["members"][position]:
                    assert drawn["after"][slot] == drawn["before"][slot]

    def test_a_replacement_never_comes_from_the_bags_own_class(self):
        """A contaminant of the bag's own class would corrupt nothing at all."""
        drawn = _material(self.RATE)
        for entry in drawn["record"]["bags"]:
            assert entry["label"] not in entry["donorClasses"]

    def test_no_single_class_donates_the_whole_contamination(self):
        """An unbalanced draw would build a confusion pair nobody declared."""
        drawn = _material(self.RATE)
        donated = {}
        for entry in drawn["record"]["bags"]:
            for donor in entry["donorClasses"]:
                donated[donor] = donated.get(donor, 0) + 1
        total = sum(donated.values())
        assert total > 0
        # Balanced across the nine classes that are never the bag's own; the
        # slack is the remainder of that division, not a tolerance chosen to pass.
        for count in donated.values():
            assert count <= -(-total // (config.CLASSES - 1)) + config.BAGS_PER_CLASS

    def test_the_record_names_every_slot_it_moved(self):
        """A manifest that under-reports is worse than one that is absent."""
        drawn = _material(self.RATE)
        for entry in drawn["record"]["bags"]:
            slots = drawn["members"][entry["bag"]]
            for offset, image in zip(entry["slots"], entry["imageIndices"]):
                assert drawn["after"][slots[offset]] == image
            assert len(entry["slots"]) == len(set(entry["slots"]))


class TestTheCap:
    """Past the cap the bag's own class stops being the plurality."""

    def test_a_rate_at_or_above_one_half_is_refused(self):
        with pytest.raises(ValueError):
            config.noise_instances(config.NOISE_CAP)

    def test_every_declared_level_sits_under_the_cap_and_is_exact(self):
        for rate in config.NOISE_LEVELS:
            count = config.noise_instances(rate)
            assert count == rate * config.INSTANCES_PER_BAG
            assert count * 2 < config.INSTANCES_PER_BAG or rate == 0.0

    def test_the_reported_level_is_the_midpoint_and_not_a_choice(self):
        """Chosen after the curve it would be the level that flatters the method."""
        assert config.NOISE_REPORTED == config.NOISE_LEVELS[len(config.NOISE_LEVELS) // 2]


class TestTheRateReachesTheRefusal:
    """The join between the stamp and the declaration, crossed rather than assumed.

    Testing each half against a fixture written by the same hand proves both
    halves and never the connection, which is the only thing the rule was about:
    `identicalAcrossShards` naming a field that never reaches a stamp checks
    nothing at all, and does it silently. So the stamp is written by
    `write_shard_stamp` and read by `disagreements` through the declaration the
    package actually ships.
    """

    def _stamp(self, tmp_path, monkeypatch, rate, name):
        """A real stamp, written by the function the campaign writes it with.

        Hand-rolling the dict here would be writing the fixture this test exists
        to avoid: the whole point is that the field survives the trip through
        `write_shard_stamp` rather than only through my own idea of it.
        """
        from dataclasses import replace
        from MIL_CREDA_Benchmark import harness
        import json

        # PRODUCT además de RESULTS: `results_for` en una tasa distinta de cero
        # cuelga de PRODUCT, así que parchar sólo RESULTS dejaba los sellos
        # contaminados cayendo dentro del repositorio de verdad. Lo hizo: quedaron
        # `Results/Noise/rho0p2/shards/s00` y `s01` en el árbol real.
        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        reduction = replace(harness.Reduction(), labelNoise=rate)
        path = harness.write_shard_stamp(name, reduction)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_rate_is_a_field_disagreements_can_actually_see(
            self, tmp_path, monkeypatch):
        """Flat and top-level: `disagreements` reads `stamp.get(field)`, never a
        dotted path, so a nested rate would be named in the declaration and
        checked nowhere."""
        from MIL_CREDA_Benchmark import shards

        stamp = self._stamp(tmp_path, monkeypatch, 0.2, "s00")
        assert stamp.get("labelNoise") == 0.2
        assert "labelNoise" in shards.declaration()["identicalAcrossShards"]

    def test_two_shards_at_different_rates_are_refused_by_name(
            self, tmp_path, monkeypatch):
        """Not averaged, refused. A different rate is a different experiment
        rather than different hardware, and one table over both is a table
        nobody can attribute."""
        from MIL_CREDA_Benchmark import shards

        clean = self._stamp(tmp_path, monkeypatch, 0.0, "s00")
        dirty = self._stamp(tmp_path, monkeypatch, 0.2, "s01")
        entries = [{"shard": "s00", "stamp": clean},
                   {"shard": "s01", "stamp": dirty}]
        clashes = shards.disagreements(
            entries, shards.declaration()["identicalAcrossShards"])
        assert [c["field"] for c in clashes] == ["labelNoise"]

    def test_two_shards_at_the_same_rate_raise_nothing(
            self, tmp_path, monkeypatch):
        """The other pole. A check that fired on agreement too would be refusing
        the split rather than the mismatch."""
        from MIL_CREDA_Benchmark import shards

        entries = [{"shard": "s00", "stamp": self._stamp(tmp_path, monkeypatch, 0.2, "s00")},
                   {"shard": "s01", "stamp": self._stamp(tmp_path, monkeypatch, 0.2, "s01")}]
        clashes = shards.disagreements(
            entries, shards.declaration()["identicalAcrossShards"])
        assert [c["field"] for c in clashes] == []


class TestTwoRatesDoNotShareATree:
    """A contaminated campaign writes elsewhere, and a clean one never moves.

    `runs.jsonl` is opened `"w"` and truncated on every campaign, so a shared
    tree would have the second rate destroy the first silently -- before
    `identicalAcrossShards` ever got the chance to refuse anything.
    """

    def test_the_clean_tree_is_exactly_where_it_has_always_been(self):
        """Every notebook, every record already on disk and the `records` block
        of the declaration name these paths. The axis is an addition."""
        assert config.results_for(0.0) == config.RESULTS
        assert config.results_for(0) == config.RESULTS

    def test_a_contaminated_rate_gets_a_tree_of_its_own(self):
        clean = config.results_for(0.0)
        for rate in config.NOISE_LEVELS[1:]:
            assert config.results_for(rate) != clean

    def test_no_two_rates_land_in_the_same_place(self):
        seen = {config.results_for(rate) for rate in config.NOISE_LEVELS}
        assert len(seen) == len(config.NOISE_LEVELS)

    def test_the_run_file_follows_the_rate_and_not_only_the_shard(self):
        from MIL_CREDA_Benchmark import harness

        clean = harness.shard_paths(None)["runs"]
        dirty = harness.shard_paths(None, noise=config.NOISE_REPORTED)["runs"]
        assert clean != dirty
        assert clean == harness.shard_paths(None, noise=0.0)["runs"]

    def test_a_sharded_run_follows_the_rate_too(self):
        """The per-shard branch has its own root expression, so it needs its own
        assertion: a shard writing into the clean tree while its stamp says 0.2
        is the mismatch the merge refusal cannot see, because it never gets two
        stamps to compare."""
        from MIL_CREDA_Benchmark import harness

        clean = harness.shard_paths("s00")["stamp"]
        dirty = harness.shard_paths("s00", noise=config.NOISE_REPORTED)["stamp"]
        assert clean != dirty


class TestTheAxisReadsWhatRan:
    """The reader and the renderers, against a tree that actually holds records."""

    METRIC = "targetAccuracy"

    def _tree(self, tmp_path, monkeypatch, values, kind="curve"):
        """One results tree per rate, written the way a run of that kind writes it.

        `kind` is a parameter and not a default buried in the helper: the sweep
        and the campaign are different shapes at the same rate, and a fixture
        that wrote one while the reader read the other would test nothing and
        pass.
        """
        import json

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        for rate, per_arm in values.items():
            root = config.results_for(rate, kind)
            root.mkdir(parents=True, exist_ok=True)
            lines = []
            for arm, value in per_arm.items():
                for transfer in ("M->U", "U->M"):
                    lines.append(json.dumps({"arm": arm, "transfer": transfer,
                                             self.METRIC: value}))
            (root / "runs.jsonl").write_text("\n".join(lines), encoding="utf-8")
            (root / "summary.json").write_text(
                json.dumps({"reduction": {"labelNoise": rate}, "grid": {}}),
                encoding="utf-8")

    def test_only_the_levels_that_ran_are_reported_as_available(
            self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import contamination as axis

        self._tree(tmp_path, monkeypatch, {0.0: {"B": 0.8}, 0.2: {"B": 0.6}})
        assert axis.available() == [0.0, 0.2]
        assert axis.missing() == [0.1, 0.3, 0.4]

    def test_an_arm_absent_from_one_level_is_named_and_never_drawn(
            self, tmp_path, monkeypatch):
        """A series with a hole beside a complete one differs in how many points
        it carries, and the eye reads density as coverage."""
        from MIL_CREDA_Benchmark import contamination as axis

        self._tree(tmp_path, monkeypatch,
                   {0.0: {"B": 0.8, "G": 0.9}, 0.2: {"B": 0.6}})
        drawn = axis.curve(self.METRIC)
        assert drawn["arms"] == ["B"]
        assert drawn["dropped"] == ["G"]

    def test_the_table_says_which_levels_never_ran(self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import tables

        self._tree(tmp_path, monkeypatch, {0.0: {"B": 0.8}, 0.2: {"B": 0.6}})
        rendered = tables.render_noise(self.METRIC, markdown=True)
        assert "ρ=0.1" in rendered and "ρ=0.4" in rendered

    def test_a_directory_that_contradicts_its_own_record_is_visible(
            self, tmp_path, monkeypatch):
        """A directory name is not evidence. The rate that governs a table is the
        one the campaign stamped into its own bounds."""
        import json
        from MIL_CREDA_Benchmark import contamination as axis

        self._tree(tmp_path, monkeypatch, {0.2: {"B": 0.6}})
        root = config.results_for(0.2, "curve")
        (root / "summary.json").write_text(
            json.dumps({"reduction": {"labelNoise": 0.3}, "grid": {}}),
            encoding="utf-8")
        assert axis.mismatched(axis.load(0.2)) is True

    def test_the_conclusion_names_who_falls_least_and_who_falls_most(
            self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import tables

        self._tree(tmp_path, monkeypatch,
                   {0.0: {"A": 0.80, "G": 0.80}, 0.4: {"A": 0.40, "G": 0.75}})
        said = tables.conclusion_noise(self.METRIC)
        assert config.NAME_OF["G"] in said and config.NAME_OF["A"] in said

    def test_the_conclusion_is_not_tied_to_nothing(self, tmp_path, monkeypatch):
        """Permute the record and the sentence has to change. A conclusion that
        comes out the same whatever the numbers say is measuring nothing, for the
        same reason an assertion that cannot fail proves nothing."""
        from MIL_CREDA_Benchmark import tables

        self._tree(tmp_path, monkeypatch,
                   {0.0: {"A": 0.80, "G": 0.80}, 0.4: {"A": 0.40, "G": 0.75}})
        one = tables.conclusion_noise(self.METRIC)
        self._tree(tmp_path, monkeypatch,
                   {0.0: {"A": 0.80, "G": 0.80}, 0.4: {"A": 0.75, "G": 0.40}})
        other = tables.conclusion_noise(self.METRIC)
        assert one != other

    def test_the_second_conclusion_says_what_the_first_cannot(
            self, tmp_path, monkeypatch):
        """It reports the difference between the two rates, which neither table
        contains on its own -- rather than enumerating the table beside it."""
        from MIL_CREDA_Benchmark import tables

        self._tree(tmp_path, monkeypatch,
                   {0.0: {"A": 0.80, "G": 0.80}, 0.2: {"A": 0.50, "G": 0.78}},
                   kind="campaign")
        said = tables.conclusion_versus_clean(self.METRIC, 0.2)
        assert "0.2" in said
        assert config.NAME_OF["G"] in said and config.NAME_OF["A"] in said

    def test_every_new_section_states_what_it_aims_at(self):
        """`verify` reports a section that never states its objective as
        `unaimed`, and a direction is not a target."""
        from MIL_CREDA_Benchmark import tables

        for key in ("noise", "noise.share", "noise.diagnostic"):
            assert len(tables.objective(key)) > 80


class TestTheSweepAndTheCampaignAreNotOneRun:
    """One transfer across every level, and every transfer at one level.

    Same rate, different shape. `runs.jsonl` is opened `"w"`, so sharing a tree
    would have whichever ran second truncate the first in silence -- the exact
    failure `results_for` was built for, one level further down.
    """

    def test_no_rate_puts_the_two_shapes_in_one_place(self):
        for rate in config.NOISE_LEVELS:
            assert config.results_for(rate, "campaign") != config.results_for(rate, "curve")

    def test_the_run_files_differ_at_every_rate(self):
        from MIL_CREDA_Benchmark import harness

        for rate in config.NOISE_LEVELS:
            campaign = harness.shard_paths(None, noise=rate, kind="campaign")["runs"]
            curve = harness.shard_paths(None, noise=rate, kind="curve")["runs"]
            assert campaign != curve

    def test_an_unknown_kind_is_refused_rather_than_guessed(self):
        """A typo resolving to some default would write a real run somewhere
        nobody reads, and report nothing."""
        import pytest

        with pytest.raises(ValueError):
            config.results_for(0.2, "campain")

    def test_the_axis_reader_defaults_to_the_sweep(self):
        """The degradation curve is what the axis is for; a default pointing at
        the campaign would draw six transfers into a one-transfer curve."""
        from MIL_CREDA_Benchmark import contamination as axis

        assert axis.KIND == "curve"
        # Con `pilot` explícito, para preguntar por la FORMA y no por cuál de
        # los dos árboles existe: sin él `level_dir` contesta «la que rige», que
        # es lo correcto para un lector y no lo que esta prueba está midiendo.
        assert axis.level_dir(0.2, pilot=False) == config.results_for(0.2, "curve")
        assert axis.level_dir(0.2, pilot=True) == config.results_for(0.2, "curve", True)


class TestThePilotWritesWhereTheRealRunDoesNot:
    """Un ensayo escribe todo lo que escribe una corrida completa, en su propio árbol.

    Compartir destino tiene una dirección peor que la otra: el que sobrescribe
    es el barato. Una campaña de horas borrada por un piloto de minutos es la
    pérdida que ninguna de estas separaciones puede reparar después.
    """

    def test_no_shape_of_pilot_lands_where_its_real_run_lands(self):
        for rate in config.NOISE_LEVELS:
            for kind in ("campaign", "curve"):
                assert (config.results_for(rate, kind, pilot=True)
                        != config.results_for(rate, kind, pilot=False))
            assert config.models_for(rate, pilot=True) != config.models_for(rate, pilot=False)

    def test_the_pilot_keeps_the_shape_and_moves_only_the_root(self):
        """Un piloto que además reordenara sus archivos no sería el mismo
        programa que la corrida que dice ensayar."""
        for rate in config.NOISE_LEVELS:
            for kind in ("campaign", "curve"):
                real = config.results_for(rate, kind, pilot=False)
                ensayo = config.results_for(rate, kind, pilot=True)
                assert ensayo.name == real.name

    def test_two_pilot_shapes_never_share_a_place_either(self):
        vistos = {config.results_for(r, k, pilot=True)
                  for r in config.NOISE_LEVELS for k in ("campaign", "curve")}
        assert len(vistos) == len(config.NOISE_LEVELS) * 2

    def test_the_run_file_and_the_stamp_follow_the_pilot_flag(self):
        from MIL_CREDA_Benchmark import harness

        for key in ("runs", "stamp"):
            assert (harness.shard_paths(None, pilot=True)[key]
                    != harness.shard_paths(None, pilot=False)[key])

    def test_the_bounds_carry_it_so_the_record_says_what_it_is(self, tmp_path,
                                                               monkeypatch):
        """Un registro que no dice que es de ensayo es exactamente cómo un
        número de piloto termina citado como resultado."""
        import json
        from dataclasses import replace
        from MIL_CREDA_Benchmark import harness

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        path = harness.write_shard_stamp(
            None, replace(harness.Reduction(), pilot=True))
        assert json.loads(path.read_text(encoding="utf-8"))["pilot"] is True


class TestTheFullRunWinsAndThePilotSaysSoWhenItDoesNot:
    """Qué registro rige cuando existen los dos, y que el informe lo diga.

    La corrida completa gana siempre. Que un ensayo le ganara haría que un
    informe cambiara de fuente sin que nadie lo tocara, y hacia abajo: números
    que no se pueden citar desplazando a los que sí. Que el completo tape al
    ensayo es la dirección segura --- lo peor que pasa es que se muestre lo bueno.
    """

    def _escribir(self, tmp_path, monkeypatch, pilot, valor):
        import json

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        root = config.results_for(0.0, "campaign", pilot)
        root.mkdir(parents=True, exist_ok=True)
        (root / "runs.jsonl").write_text(
            json.dumps({"arm": "B", "transfer": "M->U", "targetAccuracy": valor}),
            encoding="utf-8")
        (root / "summary.json").write_text(
            json.dumps({"reduction": {"labelNoise": 0.0, "pilot": pilot,
                                      "epochs": 3, "seeds": [0]}}),
            encoding="utf-8")
        return root

    def test_with_nothing_at_all_it_says_so_rather_than_returning_empty(
            self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import contamination as axis

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        assert axis.in_force(0.0) is None
        assert "Sin registro" in axis.source_note(None, 0.0)

    def test_only_the_pilot_is_used_and_announced_as_a_pilot(
            self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import contamination as axis

        self._escribir(tmp_path, monkeypatch, True, 0.5)
        level = axis.in_force(0.0)
        assert level["pilot"] is True
        assert "ENSAYO" in axis.source_note(level, 0.0)

    def test_the_full_run_wins_when_both_exist(self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import contamination as axis

        self._escribir(tmp_path, monkeypatch, True, 0.5)
        self._escribir(tmp_path, monkeypatch, False, 0.9)
        level = axis.in_force(0.0)
        assert level["pilot"] is False
        assert level["runs"][0]["targetAccuracy"] == 0.9

    def test_the_note_is_written_even_when_nothing_is_wrong(self, tmp_path,
                                                            monkeypatch):
        """Un aviso que aparece sólo en el caso malo no le enseña a nadie qué es
        lo que vigila, y la primera vez que falta se lee como que no había nada
        que avisar."""
        from MIL_CREDA_Benchmark import contamination as axis

        self._escribir(tmp_path, monkeypatch, False, 0.9)
        assert axis.source_note(axis.in_force(0.0), 0.0).strip()


class TestTheContaminationReachesTheMaterializedBags:
    """La junta entre la sustitución y el material que el modelo realmente ve.

    Todo lo de arriba mide índices. `build()` es lo que los convierte en un
    tensor, y entre las dos cosas está `torch.stack` --- exactamente el punto
    donde una contaminación correcta sobre listas puede no llegar a las
    imágenes. Probar las dos mitades por separado verifica las dos mitades y no
    la conexión, que es lo único que la regla decía.

    Corre sobre MNIST, el dominio más barato de decodificar, y se salta si el
    material no está en la caché: una prueba que exige red no es una prueba.
    """

    @pytest.fixture(scope="class")
    def torch(self):
        import torch as _torch

        if not (config.DATA_CACHE / "MNIST").is_dir():
            pytest.skip("MNIST no está en la caché local")
        return _torch

    def test_a_clean_build_and_a_contaminated_one_differ_only_in_training_bags(
            self, torch):
        limpio = bags.build("M", config.DATA_CACHE, 0, noise=0.0)
        sucio = bags.build("M", config.DATA_CACHE, 0, noise=0.4)

        movidas = 0
        for position in limpio.train_idx.tolist():
            a = limpio.images[limpio.members[position]]
            b = sucio.images[sucio.members[position]]
            movidas += int((~torch.isclose(a, b).flatten(1).all(dim=1)).sum())
        assert movidas == config.noise_instances(0.4) * len(limpio.train_idx)

        for role in ("valid_idx", "eval_idx"):
            for position in getattr(limpio, role).tolist():
                assert torch.equal(limpio.images[limpio.members[position]],
                                   sucio.images[sucio.members[position]])

    def test_the_labels_never_move(self, torch):
        """Lo que se corrompe es la evidencia, no la respuesta."""
        limpio = bags.build("M", config.DATA_CACHE, 0, noise=0.0)
        sucio = bags.build("M", config.DATA_CACHE, 0, noise=0.4)
        assert torch.equal(limpio.labels, sucio.labels)

    def test_the_manifest_rebuilds_the_contaminated_material(self, torch):
        """`rebuild` no sabe nada de ruido y no tiene por qué: `imageIndices` ya
        nombra a los contaminantes. Si esto se rompe, el material contaminado
        deja de ser reproducible sin volver a confiar en una permutación."""
        sucio = bags.build("M", config.DATA_CACHE, 0, noise=0.4)
        otra_vez = bags.rebuild(sucio.manifest, config.DATA_CACHE)
        assert torch.equal(sucio.images, otra_vez.images)
        assert torch.equal(sucio.labels, otra_vez.labels)


class TestEveryWriterFollowsTheRunAndNotTheTree:
    """Todo lo que una corrida escribe cae en el árbol de esa corrida.

    Escrito después de que el piloto muriera con un `FileNotFoundError` sobre
    `Results/Benchmark/shard.json`: la campaña escribía en `Results/Pilot/...`
    y sellaba en el árbol completo. Cada escritor por separado estaba bien y la
    corrida como conjunto estaba partida en dos, que es la falla que sólo
    aparece cuando algo la recorre entera.

    Enumerado desde la firma y no desde una lista escrita a mano: un escritor
    nuevo que olvide una de las coordenadas cae acá en vez de descubrirse en
    minuto cuarenta de un piloto.
    """

    COORDENADAS = ("labelNoise", "kind", "pilot")

    def test_the_bounds_carry_all_three_coordinates(self):
        """Las tres viajan juntas o no viajan.

        Empezaron como argumentos sueltos y eso fue el defecto: mientras `kind`
        iba por su lado, la campaña escribía sus corridas en un árbol y su sello
        en otro, y cada escritor por separado estaba bien. Puestas en la
        reducción llegan juntas a todos, y un escritor nuevo las recibe sin que
        nadie se acuerde de pasárselas.
        """
        from dataclasses import fields
        from MIL_CREDA_Benchmark import harness

        declarados = {f.name for f in fields(harness.Reduction)}
        faltan = [c for c in self.COORDENADAS if c not in declarados]
        assert not faltan, f"la reducción no lleva {faltan}"

    def test_no_writer_takes_a_coordinate_the_bounds_already_carry(self):
        """Derivado del módulo y no de una lista escrita a mano.

        La versión anterior de esta prueba enumeraba `shard_paths` y
        `seal_shard_stamp`, se olvidaba de `write_shard_stamp`, y por eso pasó
        en verde mientras el piloto moría por exactamente eso. Una lista a mano
        sólo puede contener lo que alguien ya recordó.
        """
        import inspect
        from MIL_CREDA_Benchmark import harness

        culpables = {}
        for nombre, objeto in vars(harness).items():
            if not inspect.isfunction(objeto) or nombre.startswith("_"):
                continue
            firma = inspect.signature(objeto).parameters
            if "reduction" not in firma:
                continue
            sueltas = [c for c in self.COORDENADAS
                       if c.lower().replace("label", "") in
                       {p.lower() for p in firma} or c in firma]
            # Tomar una coordenada aparte es legítimo cuando se la reconcilia:
            # la búsqueda sobrescribe la tasa a propósito, porque los techos se
            # buscan en limpio incluso para una campaña contaminada. Lo que no
            # puede quedar es la coordenada viva en dos lugares a la vez, que es
            # como el sello terminó en un árbol y las corridas en otro.
            # Sin espacios: `replace(\n    reduction,` es la misma
            # reconciliación que `replace(reduction,` y buscar el literal
            # dejaba pasar la mitad de los casos por cómo quedó el sangrado.
            fuente = re.sub(r"\s+", "", inspect.getsource(objeto))
            if sueltas and "replace(reduction" not in fuente:
                culpables[nombre] = sueltas
        assert not culpables, (
            f"estos toman una coordenada Y la reducción sin reconciliarlas: dos "
            f"fuentes para un mismo destino -> {culpables}")

    def test_the_seal_lands_beside_the_runs_it_seals(self, tmp_path, monkeypatch):
        import json
        from dataclasses import replace
        from MIL_CREDA_Benchmark import harness

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        reduccion = replace(harness.Reduction(), pilot=True,
                            labelNoise=config.NOISE_REPORTED)
        sello = harness.write_shard_stamp(None, reduccion)
        corridas = harness.shard_paths(None, noise=reduccion.labelNoise,
                                       pilot=reduccion.pilot)["runs"]
        assert sello.parent == corridas.parent

        sellado = harness.seal_shard_stamp(None, noise=reduccion.labelNoise,
                                           pilot=reduccion.pilot)
        assert sellado == sello
        assert "outputs" in json.loads(sellado.read_text(encoding="utf-8"))["evidence"]

    def test_sealing_the_wrong_tree_raises_rather_than_sealing_nothing(
            self, tmp_path, monkeypatch):
        """El otro polo: si el sello no siguiera a la corrida, esto es lo que
        pasaría --- y pasó. Un archivo ausente es la forma honesta de fallar;
        sellar un árbol vacío en silencio sería peor."""
        from dataclasses import replace
        from MIL_CREDA_Benchmark import harness

        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
        harness.write_shard_stamp(None, replace(harness.Reduction(), pilot=True))
        with pytest.raises(FileNotFoundError):
            harness.seal_shard_stamp(None, pilot=False)


class TestNoTwoFunctionsShareAName:
    """Dos definiciones con el mismo nombre en un módulo no son dos funciones.

    `tables._repetitions` estaba escrito dos veces --- una leía una reducción,
    la otra lecturas de fase dos --- y la de abajo tapaba a la de arriba.
    `conclusion()` pedía la primera, recibía la segunda y moría con `string
    indices must be integers` en la conclusión de CADA métrica del informe.
    Ninguna suite lo vio: ninguna ejecuta el cuaderno, y el módulo importa
    perfecto. Lo encontró el piloto la primera vez que algo lo corrió entero.

    Derivado del árbol de sintaxis y no de una lista: una colisión nueva cae
    acá aunque nadie se acuerde de agregarla.
    """

    import_paths = ("tables", "figures", "harness", "bags", "contamination",
                    "shards", "latent", "steps", "config", "verdict")

    def test_no_module_defines_the_same_name_twice(self):
        import ast
        import collections
        from pathlib import Path

        paquete = Path(__file__).resolve().parents[1] / "src" / "MIL_CREDA_Benchmark"
        colisiones = {}
        for modulo in sorted(paquete.glob("*.py")):
            arbol = ast.parse(modulo.read_text(encoding="utf-8"))
            nombres = [n.name for n in arbol.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef))]
            repetidos = [n for n, c in collections.Counter(nombres).items() if c > 1]
            if repetidos:
                colisiones[modulo.name] = repetidos
        assert not colisiones, (
            f"una definición tapa a la otra y el módulo importa igual: "
            f"{colisiones}")

    def test_the_conclusion_that_broke_now_runs(self):
        """El polo concreto: la función que quedaba tapada era la que
        `conclusion()` necesitaba, y sin ella el informe no tenía conclusiones.

        Corría sobre `seconds` y salteaba cuando no había campaña en disco, así
        que el polo no se probaba nunca. Y `seconds` es `perRun`: el día que
        hubiera campaña la llamada se habría negado ---`conclusion` pasa por
        `cells`--- y el rojo habría parecido de otra cosa. Va sobre una
        dimensión agrupable y sobre corridas construidas acá, porque el defecto
        era del módulo y no del árbol de resultados.
        """
        from MIL_CREDA_Benchmark import tables

        runs = [{"arm": arm, "transfer": f"{source}->{target}", "seed": 0,
                 "targetAccuracy": value, "contribution": 0.1}
                for arm, value in (("A", 0.60), ("G", 0.80))
                for source, target in config.VERDICT_TRANSFERS]
        texto = tables.conclusion(runs, "targetAccuracy", {"seeds": [0]})
        assert config.NAME_OF["G"] in texto
        assert "repetición" in texto


class TestTheNoiseAxisRefusesToPoolAPerRunDimension:
    """La guarda llegaba a la familia del veredicto y no a la del ruido.

    `cells` la lleva, y por ahí pasan `table`, `render`, `conclusion`, los
    peldaños y las ganancias apareadas. La familia del ruido no pasa por ahí:
    agrega por `contamination.by_arm`, que promedia sobre transferencia y
    repetición juntas. Con eso, `conclusion_versus_clean("seconds", ρ)`
    promediaba tiempo de pared sobre transferencias, repeticiones y ---si hay
    shards--- máquinas, y después imprimía quién pierde menos. No era
    hipotético: la celda 506 de `Benchmark_Report_v1` lo llamaba con esa
    métrica.

    Cada entrada pública se prueba **con el árbol vacío**, y eso es lo que hace
    que la prueba mida algo. Todas vuelven temprano cuando no hay registro
    ---`curve` ni siquiera llega a `by_arm`, `conclusion_versus_clean` corta en
    «falta el registro»---, así que una guarda puesta sólo en el punto de
    agregación las dejaría pasar acá y la suite quedaría verde sin que nada se
    hubiera negado. Un candado más débil sobrevive a la prueba que trae datos;
    ésta sólo pasa si la negativa está antes del retorno temprano.
    """

    def _entries(self):
        """Cada entrada pública que agrupa, con la forma mínima de llamarla.

        Derivadas a mano y no del árbol de sintaxis a propósito: lo que hace
        falta acá no es «toda función con un parámetro `metric`» ---
        `render_per_run` lo tiene y existe justamente para NO agrupar--- sino
        las que reducen varias corridas a un número.
        """
        from MIL_CREDA_Benchmark import contamination as axis
        from MIL_CREDA_Benchmark import figures, tables

        return {
            "contamination.by_arm": lambda m: axis.by_arm([], m),
            "contamination.curve": lambda m: axis.curve(m),
            "contamination.degradation": lambda m: axis.degradation(m),
            "tables.render_noise": lambda m: tables.render_noise(m),
            "tables.conclusion_noise": lambda m: tables.conclusion_noise(m),
            "tables.conclusion_versus_clean":
                lambda m: tables.conclusion_versus_clean(m, config.NOISE_REPORTED),
            "tables.conclusion_weighting_under_noise":
                lambda m: tables.conclusion_weighting_under_noise(m),
            "tables.conclusion_rungs_versus_clean":
                lambda m: tables.conclusion_rungs_versus_clean(
                    m, config.NOISE_REPORTED),
            "figures.noise_curves": lambda m: figures.noise_curves(m),
        }

    def _empty_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PRODUCT", tmp_path)
        monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")

    def test_the_refused_set_comes_from_the_declaration(self):
        """Ni una lista literal acá ni una lista literal allá: quién es `perRun`
        lo dice el banco, y una segunda copia envejece en silencio."""
        from MIL_CREDA_Benchmark import pooling

        declared = pooling.per_run_dimensions()
        assert declared, "el banco no declara ninguna dimensión `perRun`"
        assert set(declared) == set(
            (__import__("MIL_CREDA_Benchmark").__benchmark__["distribution"]
             ["perRun"]))

    def test_every_public_noise_entry_refuses_every_declared_per_run_dimension(
            self, tmp_path, monkeypatch):
        from MIL_CREDA_Benchmark import pooling

        self._empty_tree(tmp_path, monkeypatch)
        declared = pooling.per_run_dimensions()
        for name, call in self._entries().items():
            for metric in declared:
                with pytest.raises(ValueError, match="perRun"):
                    call(metric)
                    pytest.fail(f"{name} agrupó `{metric}` sin negarse")

    def test_a_poolable_dimension_still_goes_through(
            self, tmp_path, monkeypatch):
        """El polo contrario. Sin él, una guarda que se negara a todo pasaría
        la prueba de arriba entera y nadie lo notaría hasta ver la tabla vacía."""
        from MIL_CREDA_Benchmark import contamination as axis
        from MIL_CREDA_Benchmark import tables

        self._empty_tree(tmp_path, monkeypatch)
        assert axis.by_arm([], "targetAccuracy") == {}
        assert isinstance(tables.render_noise("targetAccuracy"), str)


class TestTheTwoDestinationRootsAgree:
    """`results_for` and `models_for` are one run's two roots, so they take the
    same coordinates or one of them writes somewhere the other does not.

    Written after a pilot sweep overwrote the campaign's ten `M->U seed0`
    manifests: `results_for` sent every curve level to `Noise/curve/rho...`
    while `models_for`, which never received `kind`, sent the level at rate 0
    straight into the campaign's own checkpoint directory and relabelled its
    manifests `"kind": "curve"`. The weights happened to be byte-identical, so
    nothing failed and nothing was reported -- the only symptom was one string
    inside a JSON nobody opens.

    Derived from the signatures rather than asserted about them. `models_for`'s
    own docstring already claimed "Same two coordinates as `results_for`" while
    `results_for` took three, and a sentence cannot go red. The class-wide rule
    the reduction states -- "las tres tienen que llegar juntas a cada escritor"
    -- is not asserted here for every function, because `ceilings_record_for`,
    `search_record` and `run_search` take `pilot` alone on purpose: ceilings are
    searched clean even for a contaminated campaign. These two are siblings by
    construction, which is what makes their agreement checkable without a
    hand-written roster of exemptions.
    """

    DESTINATION_COORDINATES = ("rate", "kind", "pilot")

    def _coordinates(self, function):
        import inspect

        params = {p.lower() for p in inspect.signature(function).parameters}
        return {c for c in self.DESTINATION_COORDINATES if c in params}

    def test_both_roots_take_the_same_coordinates(self):
        results = self._coordinates(config.results_for)
        models = self._coordinates(config.models_for)
        assert results == models, (
            f"`results_for` takes {sorted(results)} and `models_for` takes "
            f"{sorted(models)}; one run's two roots disagreeing on a coordinate "
            "is how a curve level lands in the campaign's tree")

    def test_a_curve_keeps_its_checkpoints_out_of_the_campaign_tree(self):
        """The pole that matters, at the rate where the collision happens.

        Rate 0 is the only level where a curve and a campaign share every other
        coordinate, so it is the only one where a missing `kind` is invisible.
        """
        campaign = config.models_for(0.0, "campaign", pilot=True)
        curve = config.models_for(0.0, "curve", pilot=True)
        assert campaign != curve

    def test_the_checkpoint_root_follows_the_results_root(self):
        """Same shape, same relative destination -- the two roots move together
        or the analysis reads one run's weights beside another's records."""
        for kind in ("campaign", "curve"):
            for rate in (0.0, config.NOISE_REPORTED):
                results = config.results_for(rate, kind, pilot=True)
                models = config.models_for(rate, kind, pilot=True)
                assert results.relative_to(config.RESULTS.parent / "Pilot") \
                    == models.relative_to(config.MODELS.parent / "Pilot"), (
                    f"{kind} at {rate}: results land at {results} and weights at "
                    f"{models}; the two roots must mirror each other")


class TestAContaminatedSearchNeverWritesTheGoverningRecord:
    """The record the campaign reads is written only by the search that governs it.

    The rule is already stated one line above the writer -- "un ensayo nunca
    escribe donde va la respuesta que la campana consume" -- and it was applied
    to the `pilot` axis alone. `noise` got the same treatment nowhere, so the
    diagnostic's re-search at rate 0.4 over one transfer overwrote the clean
    six-transfer record: 189 lines gone, `labelNoise` 0.0 -> 0.4, and the
    campaign's own ceilings replaced by a number searched under contamination.
    Nothing raised. At full scale the same call destroys the real record, which
    costs nine and a half hours to reproduce.

    Derived from the arguments the writer already receives, never from a flag a
    caller has to remember to pass: a caller that forgets a flag is the same
    defect one indirection further away.
    """

    def test_a_clean_full_search_governs(self):
        from MIL_CREDA_Benchmark import harness

        assert harness.governs_the_ceilings_record(0.0, None) is True
        assert harness.governs_the_ceilings_record(
            0.0, list(config.SEARCH_TRANSFERS)) is True

    def test_a_contaminated_search_does_not_govern(self):
        from MIL_CREDA_Benchmark import harness

        for rate in config.NOISE_LEVELS[1:]:
            assert harness.governs_the_ceilings_record(rate, None) is False

    def test_a_search_over_fewer_transfers_does_not_govern(self):
        """The diagnostic's own shape: one transfer, so its ceiling describes
        one corner of a record whose every other corner it would erase."""
        from MIL_CREDA_Benchmark import harness

        assert harness.governs_the_ceilings_record(
            0.0, [config.SEARCH_TRANSFERS[0]]) is False

    def test_every_writer_of_that_record_is_guarded(self):
        """Derived from the module, so a third writer added later lands here.

        The previous shape of this rule lived in a comment beside one writer.
        A comment cannot go red, and the second writer carried the same comment
        and the same hole.
        """
        import inspect
        from MIL_CREDA_Benchmark import harness

        unguarded = []
        for name, obj in vars(harness).items():
            if not inspect.isfunction(obj) or name.startswith("_"):
                continue
            source = inspect.getsource(obj)
            if "ceilings_record_for" not in source or "write_text" not in source:
                continue
            if "governs_the_ceilings_record" not in source:
                unguarded.append(name)
        assert not unguarded, (
            f"these write the governing ceilings record without asking whether "
            f"the search that produced it governs: {unguarded}")
