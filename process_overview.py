import pandas as pd

unit = {
    r'\binarytrees' : ' (ms)',
    r'\regexredux' : ' (ms)',
    r'\somrsperf' : ' (ms)',
    r'\sws' : ' (req/s)',
}

data = pd.read_csv('summary.csv', header=None)


# Initialize a dictionary to store formatted data
formatted_data = {}

# Process each row to organize data by Benchmark Suite
for col, row in data.iterrows():
    suite = row[0]
    config = row[1]
    time = row[2]
    ci_lower = row[3]
    ci_upper = row[4]

    if config not in formatted_data:
        formatted_data[config] = {}

    formatted_data[config][suite] = f"{round(time)} \\footnotesize{{CI [{ci_lower:.2f}, {ci_upper:.2f}]}}"

# Create LaTeX table
latex_table = "\\begin{tabular}{l" + "r" * len(formatted_data) + "}\n\\toprule\n"
# latex_table += "& " + " & ".join([f"\\multicolumn{{1}}{{c}}{{{k + unit[k]}}}" for k in formatted_data.keys()]) + " \\\\\n\\midrule\n"
latex_table += "& " + " & ".join([f"\\multicolumn{{1}}{{c}}{{{k}}}" for k in formatted_data.keys()]) + " \\\\\n\\midrule\n"

# Add rows for each configuration
configs = set(suite for config in formatted_data.values() for suite in config)
for config in sorted(configs):
    latex_table += config
    for suite in sorted(formatted_data.keys()):
        value = formatted_data[suite].get(config, "-")
        latex_table += f" & {value}"
    latex_table += " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n"

with open("table_overview2.tex", "w") as f:
    f.write(latex_table)
