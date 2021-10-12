from __future__ import annotations

import itertools
import re
import typing as t
from abc import ABC, abstractmethod
from collections import Counter

import attr

# ------------------------------------------------------------------------------
#                               Utility functions
# ------------------------------------------------------------------------------
from eradiate.attrs import documented, parse_docs


def _camel_to_snake(name):
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _pipeline_steps_converter(value):
    result = []

    for i, step in enumerate(value):
        if isinstance(step, PipelineStep):
            result.append((f"{i}_{_camel_to_snake(step.__class__.__name__)}", step))
        else:
            result.append(step)

    return result


# ------------------------------------------------------------------------------
#                             Main pipeline definition
# ------------------------------------------------------------------------------


@attr.s
class Pipeline:
    """
    A simple data processing pipeline remotely inspired from scikit-learn's
    ``Pipeline`` class.
    """

    steps: t.List[t.Tuple[str, PipelineStep]] = attr.ib(
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

    _names: t.List[str] = attr.ib(factory=list, init=False, repr=False)

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
    def named_steps(self) -> t.Dict[str, PipelineStep]:
        """
        dict[str, :class:`.PipelineStep`]: A dictionary mapping names to their
            corresponding step.
        """
        return dict(self.steps)

    def add(
        self,
        name: str,
        step: PipelineStep,
        position: t.Optional[int] = None,
        before: t.Optional[str] = None,
        after: t.Optional[str] = None,
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
                raise ValueError("parameters `before` and `after` cannot both be set")
            position = self._step_index(before)

        if after is not None:
            if before is not None:
                raise ValueError("parameters `before` and `after` cannot both be set")
            position = self._step_index(after) + 1

        if position is None:
            self.steps.append((name, step))

        else:
            self.steps.insert(position, (name, step))

        self.update()

    def transform(self, x: t.Any) -> t.Any:
        """
        Apply the pipeline to a given data.

        Parameters
        ----------
        x
            Data to apply the pipeline to.

        Returns
        -------
        xt
            Processed data.
        """
        xt = x
        for _, _, transform in self._iter():
            xt = transform.transform(xt)
        return xt

    def _iter(self):
        """
        Generate (idx, name, trans) tuples from self.steps.
        """
        stop = len(self.steps)

        for idx, (name, trans) in enumerate(itertools.islice(self.steps, 0, stop)):
            yield idx, name, trans


# ------------------------------------------------------------------------------
#                             Pipeline step definitions
# ------------------------------------------------------------------------------


@attr.s
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
@attr.s
class ApplyCallable(PipelineStep):
    """
    Turn a callable into a pipeline step.
    """

    callable: t.Callable = documented(
        attr.ib(validator=attr.validators.is_callable()),
        type="callable",
        doc="Callable with signature ``f(x: Any) -> Any``.",
    )

    def transform(self, x: t.Any) -> t.Any:
        return self.callable(x)
