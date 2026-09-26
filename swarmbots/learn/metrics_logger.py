import csv
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
import gzip
import json
from pathlib import Path
from queue import Queue
import shutil
from threading import Lock, Thread
from typing import Any
import numpy as np
from loguru import logger

try:
    import wandb
    wandb_available = True
except Exception:
    wandb_available = False

from swarmbots.learn.summary_statistics import (
    SummaryStatistics,
    SummaryStatisticsFormat,
    format_summary_statistics,
    NO_DATA,
    NoData,
    combine_summary_statistics,
    compute_summary_statistics,
)

NEWLINE_KEY = '<newline>'
SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS = "_suppress_missing_console_key_warnings"
ConsoleMetricFormat = str | SummaryStatisticsFormat | None
_MAX_PENDING_BATCHES = 2


class _StopWorker:
    pass


_STOP_WORKER = _StopWorker()
_MetricsBatch = list[dict[str, Any]]


class MetricReduction(Enum):
    LAST = auto()
    MEAN_STD = auto()
    SUM = auto()
    RATE = auto()


@dataclass(frozen=True)
class BufferedMetric:
    value: Any
    reduction: MetricReduction


def last(value: Any) -> BufferedMetric:
    return BufferedMetric(value=value, reduction=MetricReduction.LAST)


def mean_std(value: Any) -> BufferedMetric:
    return BufferedMetric(value=value, reduction=MetricReduction.MEAN_STD)


def summed(value: Any) -> BufferedMetric:
    return BufferedMetric(value=value, reduction=MetricReduction.SUM)


def rate(numerator: int | float, denominator: int | float) -> BufferedMetric:
    return BufferedMetric(value=(numerator, denominator), reduction=MetricReduction.RATE)


class MetricsLogger:
    def __init__(
            self,
            log_dir: str | Path | None = None,
            filename: str = "log.csv",
            wandb_run: Any | None = None,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            wandb_tags: list[str] | None = None,
            wandb_config: dict[str, Any] | None = None,
            wandb_mode: str | None = None,
            wandb_kwargs: dict[str, Any] | None = None,
            wandb_step_key: str | None = "timesteps",
            ignore_keys_for_persistence: Collection[str] | None = None,
            console_keys: list[str] | list[
                tuple[str, str | SummaryStatisticsFormat | None]
                | tuple[str, str | SummaryStatisticsFormat | None, str]
            ] | None = None,
            buffer_size: int = 1,
    ) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")

        self.log_dir = Path(log_dir) if log_dir else None
        self.file_path = self.log_dir / filename if self.log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.writer = None
        self._csv_fieldnames: list[str] | None = None
        self._warned_missing_console_keys: set[str] = set()

        self._wandb_run = wandb_run
        self._wandb_managed_run = False
        self._wandb_step_key = wandb_step_key

        self._ignore_keys_for_persistence = set(ignore_keys_for_persistence or [])
        self._console_key_specs = self._normalize_console_keys(console_keys)
        self._buffer_size = buffer_size
        self._metrics_buffer: _MetricsBatch = []

        if self._wandb_run is None and wandb_project is not None:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                group=wandb_group,
                tags=wandb_tags,
                config=wandb_config,
                mode=wandb_mode,
                wandb_kwargs=wandb_kwargs,
            )

        self._worker_queue: Queue[_MetricsBatch | _StopWorker] = Queue(maxsize=_MAX_PENDING_BATCHES)
        self._worker_error: BaseException | None = None
        self._worker_error_lock = Lock()
        self._closed = False
        self._worker_thread = Thread(
            target=self._worker_loop,
            name="metrics-logger",
            daemon=True,
        )
        self._worker_thread.start()

    def log(self, metrics: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot log metrics after MetricsLogger.close().")
        self._raise_worker_error()

        buffered_metrics = metrics.copy()
        buffered_metrics.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._metrics_buffer.append(buffered_metrics)
        if len(self._metrics_buffer) < self._buffer_size:
            return

        self._enqueue_buffer()

    def flush(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return

        self._enqueue_buffer()
        self._worker_queue.join()
        self._raise_worker_error()

    def _enqueue_buffer(self) -> None:
        if not self._metrics_buffer:
            return

        metrics_batch = self._metrics_buffer
        self._metrics_buffer = []
        self._worker_queue.put(metrics_batch)

    def _worker_loop(self) -> None:
        while True:
            work_item = self._worker_queue.get()
            try:
                if work_item is _STOP_WORKER:
                    return
                if self._get_worker_error() is None:
                    try:
                        self._process_metrics_batch(work_item)
                    except BaseException as error:
                        self._set_worker_error(error)
            finally:
                self._worker_queue.task_done()

    def _process_metrics_batch(self, metrics_batch: _MetricsBatch) -> None:
        metrics = self._aggregate_buffered_metrics(metrics_batch)

        self._log_to_console(metrics)

        persistence_metrics = {
            k: v for k, v in metrics.items()
            if k != SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS
            and (not self._ignore_keys_for_persistence or k not in self._ignore_keys_for_persistence)
        }

        if self.file_path and persistence_metrics:
            self._log_to_csv(persistence_metrics)

        if self._wandb_run is not None and persistence_metrics:
            self._log_to_wandb(persistence_metrics)

    def _get_worker_error(self) -> BaseException | None:
        with self._worker_error_lock:
            return self._worker_error

    def _set_worker_error(self, error: BaseException) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = error

    def _raise_worker_error(self) -> None:
        error = self._get_worker_error()
        if error is not None:
            raise RuntimeError("MetricsLogger background worker failed.") from error

    @staticmethod
    def _aggregate_buffered_metrics(buffer: list[dict[str, Any]]) -> dict[str, Any]:
        values_by_key: dict[str, list[Any]] = {}
        for metrics in buffer:
            for key, value in metrics.items():
                values_by_key.setdefault(key, []).append(value)

        return {
            key: MetricsLogger._aggregate_metric_values(key, values)
            for key, values in values_by_key.items()
        }

    @staticmethod
    def _aggregate_metric_values(key: str, values: list[Any]) -> Any:
        wrapped_values = [value for value in values if isinstance(value, BufferedMetric)]
        if wrapped_values:
            if len(wrapped_values) != len(values):
                raise TypeError(f"Metric {key!r} mixes buffered and plain values within one logging window.")
            reductions = {value.reduction for value in wrapped_values}
            if len(reductions) != 1:
                raise ValueError(f"Metric {key!r} uses multiple reductions within one logging window: {reductions}")

            reduction = wrapped_values[0].reduction
            raw_values = [value.value for value in wrapped_values]
            if reduction is MetricReduction.LAST:
                return raw_values[-1]
            if reduction is MetricReduction.MEAN_STD:
                return compute_summary_statistics(raw_values)
            if reduction is MetricReduction.SUM:
                return sum(raw_values)
            if reduction is MetricReduction.RATE:
                numerator = sum(value[0] for value in raw_values)
                denominator = sum(value[1] for value in raw_values)
                if denominator <= 0:
                    raise ValueError(f"Metric {key!r} rate denominator must be > 0, got {denominator}.")
                return numerator / denominator
            raise TypeError(reduction)

        summary_statistics = [value for value in values if isinstance(value, SummaryStatistics)]
        if summary_statistics:
            if len(summary_statistics) != len(values):
                raise TypeError(f"Metric {key!r} mixes SummaryStatistics and plain values within one logging window.")
            return combine_summary_statistics(
                summary_statistics,
                combine_data=all(stats.data is not None for stats in summary_statistics),
                combine_histograms=any(stats.histogram is not None for stats in summary_statistics),
            )

        return values[-1]

    def _log_to_console(self, metrics: dict[str, Any]) -> None:
        parts = []

        for key, value, fmt in self._iter_console_metrics(metrics):
            if key == NEWLINE_KEY:
                value = value or ''
                parts.append(f'\n{value}')
            else:
                val_str = self._format_console_value(value, fmt=fmt)
                parts.append(f"<underline>{key}</underline>: {val_str}")

        if not parts:
            logger.warning('Nothing to log?')
            return

        logger.opt(colors=True).info(" | ".join(parts))

    def _log_to_csv(self, metrics: dict[str, Any]) -> None:
        csv_metrics: dict[str, Any] = {}

        for k, v in metrics.items():
            if isinstance(v, SummaryStatistics):
                csv_metrics[k + '__mean'] = self._replace_no_data(v.mean, round_ndigits=6)
                if v.std is not None:
                    csv_metrics[k + '__std'] = self._replace_no_data(v.std, round_ndigits=6)
                if v.skewness is not None:
                    csv_metrics[k + '__skew'] = self._replace_no_data(v.skewness, round_ndigits=6)
                if v.kurtosis is not None:
                    csv_metrics[k + '__kurt'] = self._replace_no_data(v.kurtosis, round_ndigits=6)
                if v.min_value is not None:
                    csv_metrics[k + '__min'] = self._replace_no_data(v.min_value, round_ndigits=6)
                if v.max_value is not None:
                    csv_metrics[k + '__max'] = self._replace_no_data(v.max_value, round_ndigits=6)

                if v.histogram is not None:
                    if v.histogram is NO_DATA:
                        csv_metrics[k + '__histogram_freqs'] = None
                        csv_metrics[k + '__histogram_edges'] = None
                    else:
                        csv_metrics[k + '__histogram_freqs'] = json.dumps([round(x, 6) for x in v.histogram.bin_frequencies])
                        csv_metrics[k + '__histogram_edges'] = json.dumps([round(x, 6) for x in v.histogram.bin_edges])

                csv_metrics[k + '__n'] = v.n
            else:
                csv_metrics[k] = v

        self._ensure_csv_writer(csv_metrics)
        assert self._csv_fieldnames is not None

        new_fieldnames = [k for k in csv_metrics if k not in self._csv_fieldnames]
        if new_fieldnames:
            self._expand_csv_schema(new_fieldnames)

        assert self.writer is not None
        self.writer.writerow(csv_metrics)
        self.file.flush()

    def _ensure_csv_writer(self, csv_metrics: dict[str, Any]) -> None:
        if self.file is not None:
            return

        file_exists = self.file_path.exists()
        existing_fieldnames = self._read_csv_header_fieldnames() if file_exists else None
        self._csv_fieldnames = (
            existing_fieldnames if existing_fieldnames is not None else list(csv_metrics.keys())
        )
        self.file = open(self.file_path, mode='a', newline='', encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self._csv_fieldnames,
            delimiter=';',
            extrasaction="ignore",
        )

        if not file_exists or self.file_path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def _expand_csv_schema(self, new_fieldnames: list[str]) -> None:
        assert self.file_path is not None
        assert self._csv_fieldnames is not None

        existing_rows: list[dict[str, Any]] = []
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            with open(self.file_path, mode='r', newline='', encoding='utf-8') as existing_file:
                reader = csv.DictReader(existing_file, delimiter=';')
                existing_rows = list(reader)

        updated_fieldnames = [*self._csv_fieldnames, *new_fieldnames]
        with open(self.file_path, mode='w', newline='', encoding='utf-8') as rewritten_file:
            rewrite_writer = csv.DictWriter(
                rewritten_file,
                fieldnames=updated_fieldnames,
                delimiter=';',
                extrasaction="ignore",
            )
            rewrite_writer.writeheader()
            if existing_rows:
                rewrite_writer.writerows(existing_rows)

        if self.file is not None:
            self.file.close()

        self._csv_fieldnames = updated_fieldnames
        self.file = open(self.file_path, mode='a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self._csv_fieldnames,
            delimiter=';',
            extrasaction="ignore",
        )
        logger.warning(f"MetricsLogger: extended CSV schema with keys: {new_fieldnames}")

    def _read_csv_header_fieldnames(self) -> list[str] | None:
        if self.file_path is None or not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return None

        with open(self.file_path, mode='r', newline='', encoding='utf-8') as existing_file:
            reader = csv.reader(existing_file, delimiter=';')
            header = next(reader, None)

        if not header:
            return None

        return [column_name.strip() for column_name in header]

    def _init_wandb(
            self,
            project: str,
            entity: str | None,
            run_name: str | None,
            group: str | None,
            tags: list[str] | None,
            config: dict[str, Any] | None,
            mode: str | None,
            wandb_kwargs: dict[str, Any] | None,
    ) -> None:
        if not wandb_available:
            logger.warning(f"wandb is not available; continuing without wandb logging.")
            return

        init_kwargs: dict[str, Any] = dict(wandb_kwargs or {})
        init_kwargs.setdefault("project", project)
        if entity is not None:
            init_kwargs.setdefault("entity", entity)
        if run_name is not None:
            init_kwargs.setdefault("name", run_name)
        if group is not None:
            init_kwargs.setdefault("group", group)
        if tags is not None:
            init_kwargs.setdefault("tags", tags)
        if config is not None:
            init_kwargs.setdefault("config", config)
        if self.log_dir is not None:
            init_kwargs.setdefault("dir", str(self.log_dir))
        if mode is not None:
            init_kwargs.setdefault("mode", mode)

        self._wandb_run = wandb.init(**init_kwargs)
        self._wandb_managed_run = True

    def _log_to_wandb(self, metrics: dict[str, Any]) -> None:
        if not wandb_available:
            logger.warning(f"wandb is not available; skipping wandb log.")
            return

        metrics = {k: v for k, v in metrics.items() if v is not None}

        wandb_metrics: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, SummaryStatistics):
                if v.data is not None and v.data.size > 0:
                    wandb_metrics[k] = wandb.Histogram(v.data, num_bins=32)
                else:
                    wandb_metrics[k + "__mean"] = self._replace_no_data(v.mean)
                    if v.std is not None:
                        wandb_metrics[k + "__std"] = self._replace_no_data(v.std)
                    if v.skewness is not None:
                        wandb_metrics[k + "__skew"] = self._replace_no_data(v.skewness)
                    if v.kurtosis is not None:
                        wandb_metrics[k + "__kurt"] = self._replace_no_data(v.kurtosis)
                    if v.min_value is not None:
                        wandb_metrics[k + "__min"] = self._replace_no_data(v.min_value)
                    if v.max_value is not None:
                        wandb_metrics[k + "__max"] = self._replace_no_data(v.max_value)

                    if v.histogram is not None and v.histogram is not NO_DATA:
                        wandb_metrics[k + "__histogram_freqs"] = v.histogram.bin_frequencies
                        wandb_metrics[k + "__histogram_edges"] = v.histogram.bin_edges
                        wandb_hist = wandb.Histogram(
                            np_histogram=(
                                np.asarray(v.histogram.bin_frequencies, dtype=np.float32),
                                np.asarray(v.histogram.bin_edges, dtype=np.float32),
                            )
                        )
                        if wandb_hist is not None:
                            wandb_metrics[k + "__histogram"] = wandb_hist

                    wandb_metrics[k + "__n"] = v.n
            else:
                wandb_metrics[k] = v

        step = None
        if self._wandb_step_key is not None:
            value = wandb_metrics.get(self._wandb_step_key)
            if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                step = int(value)

        if step is None:
            self._wandb_run.log(wandb_metrics)
        else:
            self._wandb_run.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return

        self._enqueue_buffer()
        self._worker_queue.join()
        self._worker_queue.put(_STOP_WORKER)
        self._worker_thread.join()

        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self._csv_fieldnames = None

        if self._wandb_run is not None and self._wandb_managed_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            finally:
                self._wandb_run = None
                self._wandb_managed_run = False

        self._closed = True
        self._raise_worker_error()

    def compress_persisted_log(self) -> Path | None:
        if self.file_path is None or not self.file_path.exists():
            return None

        gz_path = self.file_path.with_suffix(f"{self.file_path.suffix}.gz")
        tmp_gz_path = gz_path.with_suffix(f"{gz_path.suffix}.tmp")

        try:
            with self.file_path.open("rb") as src, gzip.open(tmp_gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp_gz_path.replace(gz_path)
            self.file_path.unlink()
        except OSError:
            logger.exception(f"Failed to compress metrics log {self.file_path.as_posix()}")
            try:
                if tmp_gz_path.exists():
                    tmp_gz_path.unlink()
            except OSError:
                logger.exception(f"Failed to remove temporary compressed log {tmp_gz_path.as_posix()}")
            return None

        logger.info(f"Compressed metrics log to {gz_path.as_posix()}")
        return gz_path

    def __del__(self) -> None:
        try:
            if hasattr(self, "_closed"):
                self.close()
        except Exception:
            pass

    def _normalize_console_keys(
        self,
            console_keys: list[str] | list[
                tuple[str, ConsoleMetricFormat]
                | tuple[str, ConsoleMetricFormat, str]
            ] | None = None
    ) -> list[tuple[str, ConsoleMetricFormat, str]] | None:
        if console_keys is None:
            return None

        items = list(console_keys)
        if not items:
            return []

        if all(isinstance(item, str) for item in items):
            key_specs: list[tuple[str, ConsoleMetricFormat, str]] = [(key, None, key) for key in items]
            return key_specs

        if all(isinstance(item, tuple) and 2 <= len(item) <= 3 for item in items):
            key_specs = []
            for key, fmt, *maybe_alias in items:
                key_specs.append((key, fmt, maybe_alias[0] if maybe_alias else key))
            return key_specs

        raise TypeError(console_keys)

    def _iter_console_metrics(self, metrics: dict[str, Any]) -> Iterable[tuple[str, Any, ConsoleMetricFormat]]:
        if self._console_key_specs is None:
            for key, value in metrics.items():
                yield key, value, None
            return

        warn_missing_console_keys = not bool(metrics.get(SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS, False))
        for key, fmt, alias in self._console_key_specs:
            if key == NEWLINE_KEY:
                yield key, fmt, None
            else:
                if key in metrics:
                    yield alias, metrics[key], fmt
                elif warn_missing_console_keys and key not in self._warned_missing_console_keys:
                    self._warned_missing_console_keys.add(key)
                    logger.warning(
                        f"MetricsLogger: console key '{key}' is missing in metrics and will be skipped."
                    )

    def _format_console_value(self, value: Any, fmt: ConsoleMetricFormat) -> str:

        if isinstance(value, SummaryStatistics):
            assert fmt is None or isinstance(fmt, SummaryStatisticsFormat), 'supply a summary statistics format'
            return format_summary_statistics(value, fmt)

        if value is not None and fmt is not None:
            try:
                return f"{value:{fmt}}"
            except Exception:
                logger.exception(f"Error formatting console value: {value} with format: {fmt}")
                return str(value)

        if isinstance(value, (int, float, np.number)):
            if np.isclose(value % 1.0, 0):
                return f"{int(value):>2}"
            return f"{value:.3f}"

        return str(value)

    @staticmethod
    def _replace_no_data(x: float | NoData, round_ndigits: int | None = None) -> float | None:
        if x is NO_DATA:
            return None
        if round_ndigits is not None:
            x = round(x, round_ndigits)
        return x
