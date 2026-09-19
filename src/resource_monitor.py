import os
import time
import threading
import psutil


class ResourceMonitor:

    def __init__(
        self,
        interval=1.0
    ):

        self.interval = interval

        self.process = psutil.Process(
            os.getpid()
        )

        self.samples = []

        self.stop_event = threading.Event()

        self.thread = None

        self.start_time = None

        self.cpu_time_start = None


    def start(self):

        self.samples = []

        self.stop_event.clear()

        self.start_time = (
            time.perf_counter()
        )

        cpu_times = (
            self.process.cpu_times()
        )

        self.cpu_time_start = (
            cpu_times.user
            + cpu_times.system
        )

        # Inicializa a medição de CPU
        self.process.cpu_percent(
            interval=None
        )

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()


    def _monitor(self):

        while not self.stop_event.wait(
            self.interval
        ):

            elapsed = (
                time.perf_counter()
                - self.start_time
            )

            cpu_percent = (
                self.process.cpu_percent(
                    interval=None
                )
            )

            memory_info = (
                self.process.memory_info()
            )

            ram_mb = (
                memory_info.rss
                / 1024
                / 1024
            )

            self.samples.append({
                "time_seconds": elapsed,
                "cpu_percent": cpu_percent,
                "ram_mb": ram_mb
            })


    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        cpu_times = (
            self.process.cpu_times()
        )

        cpu_time_end = (
            cpu_times.user
            + cpu_times.system
        )

        cpu_time = (
            cpu_time_end
            - self.cpu_time_start
        )

        return self._get_summary(
            cpu_time
        )


    def _get_summary(
        self,
        cpu_time
    ):

        if len(self.samples) == 0:

            return {
                "cpu_avg": 0,
                "cpu_peak": 0,
                "cpu_time": cpu_time,
                "ram_avg_mb": 0,
                "ram_peak_mb": 0
            }

        cpu_values = [
            sample["cpu_percent"]
            for sample in self.samples
        ]

        ram_values = [
            sample["ram_mb"]
            for sample in self.samples
        ]

        return {
            "cpu_avg": (
                sum(cpu_values)
                / len(cpu_values)
            ),

            "cpu_peak": max(
                cpu_values
            ),

            "cpu_time": cpu_time,

            "ram_avg_mb": (
                sum(ram_values)
                / len(ram_values)
            ),

            "ram_peak_mb": max(
                ram_values
            )
        }