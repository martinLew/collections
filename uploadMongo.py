# coding: utf-8
import pymongo as po
import cPickle as cp

if __name__ == "__main__":
    m = po.MongoClient()
    db = m['futures']
    col = db['future_yinhe4']
    print "prepare to load data"
    dlist = cp.load(open('./IF_main_mongo', 'rb'))
    print "data loaded"
    print "inserting to mongodb"
    col.insert_many(dlist)
    print "insert to mongodb done"
    maincontract = cp.load(open('./main_contract', 'rb'))
    col2 = db['main_contracts']
    col2.insert_many(maincontract)
    print "main contracts index done"
