import pandas as pd
import numpy as np

def create_csv_files(size, min_weight, max_weight, min_value, max_value):
    # Instance 1: All same weight, different values
    data1 = {
        'Objet': list(range(size)),
        'Poids': [10] * size,  # All weights the same
        'Valeur': np.random.randint(min_value, max_value + 1, size).tolist()  # Different random values
    }
    pd.DataFrame(data1).to_csv('instances/instance1.csv', index=False)

     # Instance 2: Same value, different weights
    if size > (max_weight - min_weight + 1):
        raise ValueError("Size exceeds the range of unique weights.")

    weights2 = np.random.choice(np.arange(min_weight, max_weight + 1), size, replace=False).tolist()
    data2 = {'Objet': list(range(size)), 'Poids': weights2, 'Valeur': [min_value] * size}  # Same value
    pd.DataFrame(data2).to_csv('instances/instance2.csv', index=False)

    # Instance 3: Unique weights and unique values
    weights3 = np.random.randint(min_weight, max_weight + 1, size).tolist()
    values3 = np.random.randint(min_value, max_value + 1, size).tolist()
    data3 = {'Objet': list(range(size)), 'Poids': weights3, 'Valeur': values3}
    pd.DataFrame(data3).to_csv('instances/instance3.csv', index=False)

    # Instance 4: Increasing weights and values
    weights4 = [min_weight + (max_weight - min_weight) * i // (size - 1) for i in range(size)]
    values4 = [min_value + (max_value - min_value) * i // (size - 1) for i in range(size)]
    data4 = {'Objet': list(range(size)), 'Poids': weights4, 'Valeur': values4}
    pd.DataFrame(data4).to_csv('instances/instance4.csv', index=False)
