import pytest

from eradiate.scenes.measure._pipeline import ApplyCallable, Pipeline, _camel_to_snake


def test_pipeline_camel_to_snake():
    assert _camel_to_snake("SomeKindOfThing") == "some_kind_of_thing"


def test_pipeline_step_apply_callable():
    f = lambda x: sum(x)

    step = ApplyCallable(f)
    assert step.transform([1, 1]) == 2


def test_pipeline_basic():
    f = lambda x: sum(x)
    g = lambda x: x + 1

    # Init from list of steps
    pipeline = Pipeline([ApplyCallable(f), ApplyCallable(g)])
    assert pipeline.transform([1, 2]) == 4
    assert list(pipeline.named_steps.keys()) == ["0_apply_callable", "1_apply_callable"]
    assert pipeline._names == ["0_apply_callable", "1_apply_callable"]

    # Init with explicit naming
    pipeline = Pipeline([("sum", ApplyCallable(f)), ("add_1", ApplyCallable(g))])
    assert pipeline.transform([1, 2]) == 4
    assert list(pipeline.named_steps.keys()) == ["sum", "add_1"]
    assert pipeline._names == ["sum", "add_1"]


def test_pipeline_add():
    sum_ = ApplyCallable(lambda x: sum(x))
    zero = ApplyCallable(lambda x: 0)
    add_1 = ApplyCallable(lambda x: x + 1)

    # Insert by index
    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    pipeline.add("zero", zero, position=1)
    assert pipeline.transform([1, 2]) == 1

    # Insert before
    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    pipeline.add("zero", zero, before="add_1")
    assert pipeline.transform([1, 2]) == 1

    # Insert after
    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    pipeline.add("zero", zero, after="add_1")
    assert pipeline.transform([1, 2]) == 0

    # Before/after override position
    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    pipeline.add("zero", zero, before="add_1", position=0)
    assert pipeline.transform([1, 2]) == 1

    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    pipeline.add("zero", zero, after="add_1", position=0)
    assert pipeline.transform([1, 2]) == 0

    # Before + after raises
    pipeline = Pipeline([("sum", sum_), ("add_1", add_1)])
    with pytest.raises(ValueError):
        pipeline.add("zero", zero, before="add_1", after="sum")
