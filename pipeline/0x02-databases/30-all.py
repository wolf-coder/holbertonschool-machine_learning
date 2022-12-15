#!/usr/bin/env python3
"""
Listing all documents in a collection:
"""


def list_all(mongo_collection):
    """
    - Python function that lists all documents in a collection:
      + Prototype: def list_all(mongo_collection):
      + Return an empty list if no document in the collection
      + mongo_collection will be the pymongo collection object
    """
    data = mongo_collection.find()
    return [doc for doc in data]
