def export_latex(df, filename):

    latex = df.to_latex(

        float_format="%.2f",

        escape=False,

        column_format="lc"

    )

    with open(filename,"w",encoding="utf8") as f:

        f.write(latex)