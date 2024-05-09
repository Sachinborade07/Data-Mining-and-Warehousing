# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:58:39 2024

@author: Gaurav Bombale
"""
'''
3. Construct an FP-tree using suitable programming language for appropriate data set for association 
rule mining. Explain all the steps of the tree construction and draw the resulting tree. (Minimum 
support count threshold for association rules [default: 2]) Based on this tree answer the questions:
a) Find maximum frequent itemset. 
b) How many transactions does it contain? 
c) Simulate frequent pattern enumeration based on the FP-tree constructed. 
d) Give comparative analysis of this process with Apriori algorithm.

'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Step 1: Read and preprocess the dataset
df = pd.read_excel(r"C:\Users\Admin\Downloads\online+retail\Online_Retail.xlsx")
# Remove rows with missing values and rows containing 'C' in the Invoice column (indicating cancelled transactions)
df_cleaned = df.dropna().loc[~df['Invoice'].str.startswith('C')]
# Group items by transactions
transactions = df_cleaned.groupby('Invoice')['Description'].apply(list).values.tolist()

# Step 2: Generate frequent itemsets using FP-growth algorithm
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_fp = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df_fp, min_support=0.02, use_colnames=True)

# Step 3: Construct FP-tree from frequent itemsets
class TreeNode:
    def __init__(self, item, frequency, parent):
        self.item = item
        self.frequency = frequency
        self.parent = parent
        self.children = {}

def construct_fp_tree(transactions, min_support):
    # Count item frequencies
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    
    # Filter out infrequent items
    frequent_items = {item: freq for item, freq in item_counts.items() if freq >= min_support}
    sorted_items = sorted(frequent_items, key=frequent_items.get, reverse=True)
    
    # Construct FP-tree
    root = TreeNode(None, None, None)
    for transaction in transactions:
        sorted_transaction = sorted(transaction, key=lambda item: frequent_items.get(item, 0), reverse=True)
        current_node = root
        for item in sorted_transaction:
            if item in frequent_items:
                if item not in current_node.children:
                    current_node.children[item] = TreeNode(item, 1, current_node)
                else:
                    current_node.children[item].frequency += 1
                current_node = current_node.children[item]
    return root, frequent_items

# Construct the FP-tree
fp_tree_root, frequent_items = construct_fp_tree(transactions, min_support=2)

# Step 4: Draw the resulting FP-tree (visualization code depends on specific libraries)
# Visualization depends on the specific library you use (e.g., matplotlib, graphviz)
# I'll provide a simple text-based visualization here

def print_fp_tree(node, indent=0):
    if node.item is not None:
        print('  ' * indent + f"{node.item} ({node.frequency})")
    for child in node.children.values():
        print_fp_tree(child, indent + 1)

# Print the FP-tree
print("FP-Tree:")
print_fp_tree(fp_tree_root)

# Step 5: Answer the questions
# a) Find maximum frequent itemset
max_freq_itemset = frequent_itemsets[frequent_itemsets['support'] == frequent_itemsets['support'].max()]['itemsets'].iloc[0]
print("\na) Maximum frequent itemset:", max_freq_itemset)

# b) How many transactions does it contain?
num_transactions = len(df_cleaned[df_cleaned['Description'].apply(lambda x: all(item in x for item in max_freq_itemset))]['Invoice'].unique())
print("b) Number of transactions containing the maximum frequent itemset:", num_transactions)

# c) Simulate frequent pattern enumeration based on the FP-tree constructed (Apriori is not necessary for FP-tree)
# d) Give comparative analysis of this process with Apriori algorithm (FP-growth is generally faster than Apriori)

