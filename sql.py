import sqlite3
import numpy as np
def creatTable(tablename):
    conn = sqlite3.connect('DATA.db')
    try:
        conn.execute('''CREATE TABLE %s
            (id        INTEGER    PRIMARY KEY      NOT NULL ,
            word            CHAR         unique  NOT NULL)
            ;''' % tablename)
    except:
        conn.close()

        return
def execute(sql):
    conn = sqlite3.connect('DATA.db')
    conn.execute(sql)
    conn.commit()
    conn.close()
def delTable(tablename,id=0,all=True):
    if (all):
        sql='''DELETE FROM %s WHERE ID >= 1 ;'''%tablename
        execute(sql)
        sql='''UPDATE sqlite_sequence SET seq = 0 WHERE name = '%s';'''%tablename
        execute(sql)
    sql='''DELETE FROM %s WHERE ID = %d;''' % (tablename,id)
    execute(sql)
def addtoSome(tablename,Key,Value):
    try:
        addSegment(tablename,Key)

    finally:
        sql = '''INSERT INTO %s ('%s')
                VALUES ('%s')''' % (tablename,Key,Value)
        execute(sql)
def addtoItem(tablename,name):

    sql = '''INSERT INTO %s (NAME)
            VALUES ('%s')''' % (tablename,name)
    execute(sql)
def addtoSegment(tablename,segmeng,data,ID):
    sql='''update %s set "%s"="%s" where ID=%d;''' % (tablename,segmeng,data,ID)
    execute(sql)
def selectId(tablename,id):
    sql ='''SELECT * FROM %s
            WHERE ID = %d''' % (tablename,id)
    conn = sqlite3.connect('DATA.db')
    cur=conn.cursor()
    cur.execute(sql)
    results=cur.fetchone()
    cur.close()
    conn.close()
    return results
def toZero():
    sql = '''DELETE FROM sqlite_sequence;'''
    execute(sql)
def addSegment(tablename,segname):
    sql='''ALTER TABLE %s ADD "%s" TEXT''' % (tablename,segname)
    execute(sql)
def copySegment(Tablename):
    sql='''CREATE TABLE TEMPTABLE
        (ID            INTEGER      PRIMARY KEY    AUTOINCREMENT  NOT NULL,
        NAME           TEXT    NOT NULL);
        insert into TEMPTABLE(ID,NAME) select ID,NAME from %s;
    '''% (Tablename)
def addData(tablename,data):
    try:
        sql = '''INSERT INTO %s (word)
                VALUES ('%s')''' % (tablename, data)
        execute(sql)
    except:
        return False

def selectData(tablename,data):
    try:
        sql = '''SELECT id FROM %s where word=('%s')''' % (tablename, data)
        conn = sqlite3.connect('DATA.db')
        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchone()
        cur.close()
        conn.close()
        return  int(results[0])
    except:
        return False
if __name__ == '__main__':
    np.set_printoptions(threshold='nan')

    creatTable('data')
    # mystr = "100110"
    # print np.array(list(mystr))
    # print np.array(list(mystr)).tostring()
    # arr = np.zeros( (10,10) )
    # print arr
    # n = np.load('pupu.npy')
    # print n[0]
    # a=[]
    # a[selectData('data','we')]=1
    # print a