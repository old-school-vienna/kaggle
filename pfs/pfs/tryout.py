import pandas as pd

import helpers as hlp

df = pd.DataFrame({'foo': ['one', 'one', 'two', 'two', 'two', 'two'],
                   'bar': ['A', 'A', 'A', 'A', 'B', 'C'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})

df1 = hlp.pivot(df, ['foo', 'zoo'], 'bar', 'baz')
print(df1)
