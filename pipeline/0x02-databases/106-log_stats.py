#!/usr/bin/env python3
"""
- Improve 34-log_stats.py by adding the top 10 of the most present
IPs in the collection nginx of the database logs:
  + The IPs top must be sorted
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school = client.logs.nginx
    print('{} logs'.format(school.count_documents({})))
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print('Methods:')
    for method in methods:
        print('\tmethod {}: {}'.format(
            method,
            school.count_documents({'method': method})
        ))
    print('{} status check'.format(
        school.count_documents(
            {'method': 'GET', 'path': '/status'}
        )
    ))
    pipeline = [
        {"$sortByCount": '$ip'},
        {"$limit": 10},
        {"$sort": {"ip": -1}},
    ]
    IPs = school.aggregate(pipeline=pipeline)
    print("IPs:")
    for ip in IPs:
        print('\t{}: {}'.format(ip['_id'], ip['count']))
