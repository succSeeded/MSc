from pymorphy2 import MorphAnalyzer

m = MorphAnalyzer()

print(m.parse("Школа")[0])
