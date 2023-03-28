from __future__ import annotations

import itertools
import logging
import typing as t
from abc import ABC, abstractmethod
from collections import Counter

import attrs

from ..attrs import documented, parse_docs
from ..util.misc import camel_to_snake

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#                                Local utilities
# ------------------------------------------------------------------------------


def _pipeline_steps_converter(value):
    result = []

    for i, step in enumerate(value):
        if isinstance(step, PipelineStep):
            result.append((f"{i}_{camel_to_snake(step.__class__.__name__)}", step))
        else:
            result.append(step)

    return result


# ------------------------------------------------------------------------------
#                          Basic classes and interfaces
# ------------------------------------------------------------------------------


@attrs.define
class Pipeline:
    """
    A simple data processing pipeline remotely inspired from scikit-learn's
    ``Pipeline`` class.
    """

    steps: list[tuple[str, PipelineStep]] = attrs.field(
        factory=list, converter=_pipeline_steps_converter
    )

    @steps.validator
    def _steps_validator(self, attribute, value):
        for id, step in value:
            if not isinstance(step, PipelineStep):
                raise ValueError(
                    f"while validating '{attribute.name}': step '{id}' is not "
                    "a PipelineStep instance"
                )

        # Check for name duplicates
        for name, count in Counter([name for name, _ in value]).items():
            if count > 1:
                raise ValueError(
                    f"while validating '{attribute.name}': found duplicate "
                    f"step name {name}"
                )

    _names: list[str] = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        """
        Update internal state. Should be run whenever `steps` is modified or
        mutated.
        """
        self._names = [name for name, _ in self.steps]

    def _step_index(self, name):
        return self._names.index(name)

    @property
    def named_steps(self) -> dict[str, PipelineStep]:
        """
        dict[str, :class:`.PipelineStep`]: A dictionary mapping names to their
            corresponding step.
        """
        return dict(self.steps)

    def add(
        self,
        name: str,
        step: PipelineStep,
        position: int | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        """
        Add a step to an existing pipeline.

        Parameters
        ----------
        name : str
            Name of the step to be added. If the passed name already exists, an
            exception will be raised.

        step : :class:`.PipelineStep`
            Step to be added to the pipeline.

        position : int, optional
             Index where `step` will be inserted.

        before : str, optional
            Insert `step` before the step with the name `name`. Exclusive with
            `after`.

        after : str, optional
            Insert `step` after the step with the name `name`. Exclusive with
            `before`.

        Raises
        ------
        ValueError
            If `name` maps to an already registered step.

        ValueError
            If both `before` and `after` are set.

        Notes
        -----
        * If none of `position`, `before` or `after` are set, the step will be
          appended to the pipeline.
        * If `position` and `before` (resp. `after`) are set, `before` (resp.
          `after`) takes precedence.
        """
        if name in self._names:
            raise ValueError(f"step name {name} already exists")

        if before is not None:
            if after is not None:
                raise ValueError("cannot set both 'before' and 'after' parameters")
            position = self._step_index(before)

        if after is not None:
            if before is not None:
                raise ValueError("cannot set both 'before' and 'after' parameters")
            position = self._step_index(after) + 1

        if position is None:
            self.steps.append((name, step))

        else:
            self.steps.insert(position, (name, step))

        self.update()

    def transform(
        self,
        x: t.Any,
        start: int | str | None = None,
        stop: int | str | None = None,
        stop_after: int | str | None = None,
        step: int | str | None = None,
    ) -> t.Any:
        """
        Apply the pipeline to a given data. Keyword arguments can be used to
        restrict pipeline execution to selected steps.

        Parameters
        ----------
        x
            Data to apply the pipeline to.

        start : int or str, optional
            If set, start execution at indexed step.

        stop : int or str, optional
            If set, stop execution at step preceding indexed step.

        stop_after : int or str, optional
            If set, stop execution after indexed step. Takes precedence over
            `stop`.

        step : int or str, optional
            If set, execute indexed step only. Takes precedence on all other
            step selectors.

        Returns
        -------
        xt
            Processed data.
        """
        if step is not None:
            if isinstance(step, str):
                step = self._step_index(step)

            return self.steps[step][1].transform(x)

        if isinstance(start, str):
            start = self._step_index(start)

        if isinstance(stop_after, str):
            stop_after = self._step_index(stop_after)

        if stop_after is not None:
            stop = stop_after + 1

        if isinstance(stop, str):
            stop = self._step_index(stop)

        xt = x
        for _, _, transform in self._iter(start, stop):
            xt = transform.transform(xt)
        return xt

    def _iter(self, start: int | None = None, stop: int | None = None):
        """
        Generate (idx, name, trans) tuples from self.steps.
        """
        if start is None:
            start = 0

        if stop is None:
            stop = len(self.steps)

        for idx, (name, trans) in enumerate(itertools.islice(self.steps, start, stop)):
            yield idx, name, trans

    @classmethod
    def convert(cls, x: t.Any) -> t.Any:
        if isinstance(x, (tuple, list)):
            return cls(x)

        else:
            return x


@attrs.define
class PipelineStep(ABC):
    """
    Interface for pipeline step definitions.
    """

    @abstractmethod
    def transform(self, x: t.Any) -> t.Any:
        """
        Apply the pipeline step to a given data.

        Parameters
        ----------
        x
            Data to process.

        Returns
        -------
        xt
            Processed data.
        """
        pass


@parse_docs
@attrs.define
class ApplyCallable(PipelineStep):
    """
    Turn a callable into a pipeline step.
    """

    callable: t.Callable = documented(
        attrs.field(validator=attrs.validators.is_callable()),
        type="callable",
        doc="Callable with signature ``f(x: Any) -> Any``.",
    )

    def transform(self, x: t.Any) -> t.Any:
        return self.callable(x)
