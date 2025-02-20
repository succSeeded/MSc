from pymorphy2 import MorphAnalyzer

m = MorphAnalyzer()

print(isinstance(m.parse("новые пиписьки"), list))
