#chatgpt helped creating this plot

import os
from quality_data import data

systems = ["GT", "iDisc", "CD", "GPT-4o", "Schuyler (No FT)", "Schuyler (DATM)"]
databases = ["MusicBrainz", "Magento", "AdventureWorks", "StackExchange", "TPC-E"]

def generate_tikz(dots):
    tikz = "\\begin{tikzpicture}[scale=0.5]\n"
    for dot in dots:
        tikz += f"  \\fill[{dot['color']}] ({dot['x']},{dot['y']}) circle (2pt);\n"
    tikz += "\\end{tikzpicture}"
    return tikz


def generate_latex_table(
    systems,
    databases,
    data,
    caption="Overview of System Performance on Databases",
    label="tab:sys_db_overview",
):
    # Begin table* environment
    latex = "\\begin{table*}[ht]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n\n"

    latex += "\\begin{tabular}{c"
    latex += "".join(["|c" for _ in systems])  # One '|c' per system
    latex += "}\n"
    latex += "\\toprule\n"

    latex += " & " + " & ".join(systems) + " \\\\\n"
    latex += "\\midrule\n"

    for db in databases:
        row_1 = (
            f"\\multirow{{2}}{{*}}{{\\centering {db}}}"
        )
        for sys in systems:
            dot_info = data[db][sys]["dots"]  # list of dots
            tikz_code = generate_tikz(dot_info)
            row_1 += " & " + tikz_code
        row_1 += " \\\\"

        row_2 = ""
        for idx, sys in enumerate(systems):
            if idx == 0:
                row_2 += "  "
            row_2 += " & " + str(data[db][sys]["score"])
        row_2 += " \\\\"

        latex += row_1 + "\n" + row_2 + "\n"
        latex += "\\midrule\n"
    latex = latex.rstrip("\\midrule\n") + "\\\\ \\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table*}\n"

    return latex

if __name__ == "__main__":
    latex_code = generate_latex_table(systems, databases, data)
    with open("table_of_dots.tex", "w") as f:
        f.write(latex_code)
    print("Generated table_of_dots.tex with two-row entries per database,")
    print("including merged database cells and centered database names.")
