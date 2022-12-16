#!/usr/bin/env python3
'''Sorting students'''


def top_students(mongo_collection):
    """
    - Write a Python function that returns all students
sorted by average score:
      + Prototype: def top_students(mongo_collection):
      + mongo_collection will be the pymongo collection object
      + The top must be ordered
      + The average score must be part of each item returns
with key = averageScore
    """
    docs = []
    for doc in mongo_collection.find():
        Sum = sum([s['score'] for s in doc['topics']])
        doc['averageScore'] = Sum / len(doc['topics'])
        docs.append(doc)
    return (sorted(docs, key=lambda a: a['averageScore'], reverse=True))
