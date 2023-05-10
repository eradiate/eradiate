import pytest

from eradiate.pipelines import ApplyCallable, Pipeline


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


def test_pipeline_transform():
    add_1 = ApplyCallable(lambda x: x + 1)
    times_3 = ApplyCallable(lambda x: x * 3)
    mod_7 = ApplyCallable(lambda x: x % 7)

    pipeline = Pipeline([("add_1", add_1), ("times_3", times_3), ("mod_7", mod_7)])

    # No param means we execute the whole pipeline
    assert ((3 + 1) * 3) % 7 == pipeline.transform(3)

    # Start index can be used to remove some steps
    assert (3 * 3) % 7 == pipeline.transform(3, start="times_3")
    assert pipeline.transform(3, start="times_3") == pipeline.transform(3, start=1)

    # Stop index as well
    assert (3 + 1) * 3 == pipeline.transform(3, stop="mod_7")
    assert pipeline.transform(3, stop="mod_7") == pipeline.transform(3, stop=2)

    # Stop after index acts like stop but includes its bound
    assert pipeline.transform(3, stop_after="times_3") == pipeline.transform(
        3, stop="mod_7"
    )
    assert pipeline.transform(3, stop_after=1) == pipeline.transform(3, stop=2)
    assert pipeline.transform(3, stop_after=0) == pipeline.transform(3, stop=1)

    # Step selector executes a single step
    assert 3 % 7 == pipeline.transform(3, step="mod_7")
    assert pipeline.transform(3, step="mod_7") == pipeline.transform(3, step=2)
