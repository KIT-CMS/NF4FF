import law
import luigi

from core.methods import FFMethod


DEFAULT_GROUPING = "njets"
DEFAULT_DRSR_LOSS_LIMIT = 0.1


class GroupingParameters:
    grouping = luigi.ChoiceParameter(
        default=DEFAULT_GROUPING,
        choices=(DEFAULT_GROUPING,),
    )


class SqueezingParameters:
    squeezing = luigi.OptionalFloatParameter(default=0.99)


class DRSRParameters(SqueezingParameters):
    squeezing_loss_limit = luigi.FloatParameter(
        default=DEFAULT_DRSR_LOSS_LIMIT
    )


class FFMethodParameters(DRSRParameters):
    correction = luigi.ChoiceParameter(
        default=FFMethod.DRSR_NONCLOSURE.value,
        choices=FFMethod.choices(),
    )


class WorkflowTask(law.Task):
    """Common base for workflow tasks."""
