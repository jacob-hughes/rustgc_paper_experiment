import re
import subprocess
import tempfile

import pandas as pd
from rebench.interop.adapter import (
    GaugeAdapter,
    OutputNotParseable,
    ResultsIndicatedAsInvalid,
)
from rebench.model.data_point import DataPoint
from rebench.model.measurement import Measurement


class AlloyAdapter(GaugeAdapter):
    # To be sure about how to parse the output, give custom format
    # This avoids issues with perhaps none-standard /bin/usr/time
    time_format = '"usr: %U\nwall-time: %e\n"'
    re_formatted_time = re.compile(r"^wall-time: (\d+\.\d+)")
    re_formatted_usr = re.compile(r"^usr: (\d+\.\d+)")

    _completed_time_availability_check = False
    _use_formatted_time = False
    _time_bin = None

    def __init__(self, include_faulty, executor):

        GaugeAdapter.__init__(self, include_faulty, executor)

    def _create_command(self, command):
        time = f"{self._time_bin} -f {AlloyAdapter.time_format}"
        return f"{time} {command}"

    def should_enable_premature_finalizer_optimization(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return "gcvs" in exec or "elision" in exec or "premopt-opt" in exec

    def should_enable_premature_finalizer_prevention(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return (
            "gcvs" in exec
            or "elision" in exec
            or "premopt-opt" in exec
            or "premopt-naive" in exec
        )

    def should_enable_finalizer_elision(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return "gcvs" in exec or "premopt" in exec or "elision-opt" in exec

    def setup_stats(self, run_id):
        bm = run_id.benchmark.name.lower()
        if "USE_MT" in run_id.env:
            dir = run_id.env["RESULTS_DIR"]
            out = f"{dir}/{bm}-{run_id.completed_invocations + 1}"
            run_id.env["GC_LOG_FILE"] = f"{out}.json"

    def acquire_command(self, run_id):
        self.setup_stats(run_id)
        command = run_id.cmdline_for_next_invocation()

        if not self._completed_time_availability_check:
            self._check_which_time_command_is_available()

        if "USE_HT" in run_id.env:
            dir = run_id.env["RESULTS_DIR"]
            bm = run_id.benchmark.name.lower()
            out = f"{dir}/{bm}-{run_id.completed_invocations + 1}"
            ht = run_id.env["HT_PATH"]
            command = " ".join([ht, "--record-only", "-o", out, command])
        return self._create_command(command)

    def _check_which_time_command_is_available(self):
        time_bin = "/usr/bin/time"
        try:
            formatted_output = subprocess.call(
                ["/usr/bin/time", "-f", AlloyAdapter.time_format, "/bin/sleep", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            formatted_output = 1

        if formatted_output == 1:
            try:
                formatted_output = subprocess.call(
                    [
                        "/opt/local/bin/gtime",
                        "-f",
                        AlloyAdapter.time_format,
                        "/bin/sleep",
                        "0",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if formatted_output == 0:
                    time_bin = "/opt/local/bin/gtime"
            except OSError:
                formatted_output = 1

        AlloyAdapter._use_formatted_time = formatted_output == 0
        AlloyAdapter._time_bin = time_bin
        AlloyAdapter._completed_time_availability_check = True

    def parse_data(self, data, run_id, invocation):
        iteration = 1
        data_points = []
        current = DataPoint(run_id)
        total_measure = None

        for line in data.split("\n"):
            if self.check_for_error(line):
                raise ResultsIndicatedAsInvalid(
                    "Output of bench program indicated error."
                )

            match1 = self.re_formatted_usr.match(line)
            match2 = self.re_formatted_time.match(line)
            if match1:
                usr = float(match1.group(1)) * 1000
                measure = Measurement(invocation, iteration, usr, "ms", run_id, "usr")
                current.add_measurement(measure)
            elif match2:
                time = float(match2.group(1)) * 1000
                measure = Measurement(
                    invocation, iteration, time, "ms", run_id, "total"
                )

                current.add_measurement(measure)
                data_points.append(current)
                current = DataPoint(run_id)
                iteration += 1

        if total_measure:
            current.add_measurement(total_measure)
            data_points.append(current)

        if not data_points:
            raise OutputNotParseable(data)

        return data_points


class AlloyMemAdapter(AlloyAdapter):
    def acquire_command(self, run_id):
        self.setup_stats(run_id)
        command = run_id.cmdline_for_next_invocation()
        return command
