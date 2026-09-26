from collections import defaultdict
from typing import TypeVar, Generic

from swarmbots.learn.summary_statistics import compute_summary_statistics

T = TypeVar('T')

class MetricsLists(Generic[T]):

    def __init__(self):
        self.lists: dict[str, list[T]] = defaultdict(list)

    def add(self, metrics: dict[str, T]):
        for k, v in metrics.items():
            self.lists[k].append(v)

    def get(self):
        return self.lists

    def compute_summary_statistics(
            self,
            find_min: bool = False,
            find_max: bool = False,
            make_histogram: bool | int = False,
            compute_skewness: bool = False,
            compute_kurtosis: bool = False,
            keep_data: bool = False,
            prefix: str | None = None
    ):

        return {
            k if prefix is None else prefix + k: compute_summary_statistics(
                v,
                find_min=find_min, find_max=find_max,
                compute_skewness=compute_skewness, compute_kurtosis=compute_kurtosis,
                make_histogram=make_histogram, keep_data=keep_data,
            ) for k, v in self.lists.items()
        }
