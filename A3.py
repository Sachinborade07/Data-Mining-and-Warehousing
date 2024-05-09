# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:28:30 2024

@author: Admin
"""

'''
Problem Statements:
    3. Construct an FP-tree using suitable programming language for appropriate data set for association
rule mining. Explain all the steps of the tree construction and draw the resulting tree. (Minimum
support count threshold for association rules [default: 2]) Based on this tree answer the questions:
a) Find maximum frequent itemset.
b) How many transactions does it contain?
c) Simulate frequent pattern enumeration based on the FP-tree constructed.
d) Give comparative analysis of this process with Apriori algorithm.
'''

import pandas as pd
from collections import defaultdict

class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next_node = None


def construct_FP_tree(database, min_support_count):
    # Step 1: Scan the Database and count item frequencies
    item_counts = defaultdict(int)
    for transaction in database:
        for item in transaction:
            item_counts[item] += 1

    # Sort items in descending order of frequency
    sorted_items = [item for item in sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)]

    # Step 2: Construct the FP-tree
    root = TreeNode(None, 0, None)  # Initialize root node
    header_table = {}  # Header table to store references to the latest occurrences of items
    for transaction in database:
        sorted_transaction = [item for item in sorted_items if item in transaction]
        current_node = root
        for item in sorted_transaction:
            current_node.children[item] = current_node.children.get(item, TreeNode(item, 0, current_node))
            current_node = current_node.children[item]
            current_node.count += 1
            # Update header table
            if item not in header_table:
                header_table[item] = current_node
            else:
                while header_table[item].next_node:
                    header_table[item] = header_table[item].next_node
                header_table[item].next_node = current_node

    # Step 3: Prune the FP-tree
    def prune_tree(tree_node, min_support_count):
        for item in list(tree_node.children.keys()):
            if tree_node.children[item].count < min_support_count:
                tree_node.children.pop(item)
        for child_node in tree_node.children.values():
            prune_tree(child_node, min_support_count)

    prune_tree(root, min_support_count)

    # Step 4: Mine Association Rules
    def mine_association_rules(prefix_path, header_table, min_support_count):
        for item in header_table.keys():
            new_frequent_set = prefix_path.copy()
            new_frequent_set.add(item)
            print("Frequent Itemset:", new_frequent_set)
            conditional_database = []
            node = header_table[item]
            while node:
                frequency = node.count
                parent = node.parent
                conditional_transaction = []
                while parent.item:
                    conditional_transaction.append(parent.item)
                    parent = parent.parent
                conditional_database.extend([[item] * frequency] * frequency)
                node = node.next_node
            print("Transactions containing the itemset:", conditional_database)

            # Recursively construct conditional FP-tree
            conditional_tree = construct_FP_tree(conditional_database, min_support_count)
            if conditional_tree:
                mine_association_rules(new_frequent_set, conditional_tree.header_table, min_support_count)

    print("\nConstructing FP-tree:")
    print_FP_tree(root)

    # Mine association rules starting from the root
    print("\nMining Association Rules:")
    mine_association_rules(set(), header_table, min_support_count)


def print_FP_tree(node, depth=0):
    if node:
        print(' ' * depth, node.item, ':', node.count)
        for child in node.children.values():
            print_FP_tree(child, depth + 4)


# Load the dataset from "groceries.csv" file
data = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/Groceries_dataset.csv', header=None)
transactions = [set(row.dropna().values) for _, row in data.iterrows()]

# Define minimum support count threshold
min_support_count = 100

# Construct the FP-tree and mine association rules
construct_FP_tree(transactions, min_support_count)
