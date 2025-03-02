import argparse
import csv
import subprocess
import time

import psutil


def monitor_rss(binary, args, output_file, interval):
    process = subprocess.Popen([binary] + args)
    pid = process.pid

    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["time", "mem"])
        start_time = time.time()
        try:
            while process.poll() is None:
                try:
                    rss = psutil.Process(pid).memory_info().rss
                    timestamp = time.time() - start_time
                    csvwriter.writerow([f"{timestamp:.2f}", f"{rss:.2f}"])
                    csvfile.flush()
                except psutil.NoSuchProcess:
                    print("Process terminated.")
                    break
                time.sleep(interval / 1000)
        except KeyboardInterrupt:
            print("Monitoring interrupted.")
        finally:
            if process.poll() is None:
                process.terminate()
            print(f"Process exited with return code: {process.returncode}")
            process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor RSS of a binary over time")
    parser.add_argument("binary")
    parser.add_argument("-o", "--output", required=True, help="CSV file")
    parser.add_argument(
        "--interval", type=float, default=10, help="Sampling interval in ms"
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments for the binary"
    )

    args = parser.parse_args()

    monitor_rss(args.binary, args.args, args.output, args.interval)
