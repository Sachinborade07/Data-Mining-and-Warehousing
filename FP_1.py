from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
import pandas as pd
import time

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        dataset = [line.strip().split(',') for line in file.readlines()]
    # Convert all elements to strings
    dataset = [[str(item) for item in transaction] for transaction in dataset]
    return dataset

path = 'OnlineRetail.csv'
# Step 1: Read the dataset
dataset = read_dataset(path)


# Step 2: Convert dataset to one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: FP-growth algorithm
start_time = time.time()
fpgrowth_result = fpgrowth(df, min_support=0.2, use_colnames=True)
fpgrowth_time = time.time() - start_time

# Step 4: Output the results
print("FP-growth algorithm result:")
print(fpgrowth_result)
print("FP-growth algorithm execution time:", fpgrowth_time)

# Step 5: Find maximum frequent itemset
max_itemset = fpgrowth_result[fpgrowth_result['support'] == fpgrowth_result['support'].max()]['itemsets']
print("\na) Maximum frequent itemset:")
print(max_itemset)

# Step 6: Number of transactions
num_transactions = len(dataset)
print("\nb) Number of transactions:", num_transactions)

# Step 7: Simulate frequent pattern enumeration based on the FP-tree constructed
simulated_frequent_patterns = fpgrowth_result.values.tolist()
print("\nc) Simulated frequent patterns:")
print(simulated_frequent_patterns)

# Step 8: Apriori algorithm
start_time = time.time()
apriori_result = apriori(df, min_support=0.3, use_colnames=True)
apriori_time = time.time() - start_time

# Step 9: Comparative analysis
print("\nApriori algorithm execution time:", apriori_time)
print("FP-growth algorithm execution time:", fpgrowth_time)
