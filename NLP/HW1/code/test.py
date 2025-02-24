from numpy import zeros, ones
from pandas import DataFrame

df_total = DataFrame(data=8 * ones((1, 10)), columns=["sentences", "words", "tags", "NOUN", "ADJ", "PRON", "DET", "VERB", "ADV", "NUM"])
df_corr = df_total.copy()

df_acc = df_corr.div(df_total)

print(df_acc.at[0, "NOUN"])