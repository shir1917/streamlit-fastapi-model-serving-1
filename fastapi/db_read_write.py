import time
start_time = time.time()
import sys
import platform
import re
import  pandas as pd
import numpy as np
from os import path



def timeStamped(fname, fmt='{fname} %d.%m.%Y %H.%M.xlsx'):
    import datetime
    # This creates a timestamped filename so we don't overwrite our good work
    return datetime.datetime.now().strftime(fmt).format(fname=fname)



def server_connection_mode(mode):
    if ('rerun' in sys.argv):
        mode = 'rerun'

    if (sys.platform == "Windows" or sys.platform == "win32"):
        import pyodbc
        print("running on Windows, Python version ", platform.sys.version)
        # "Driver={ODBC Driver 17 for SQL Server};"
        # server = pyodbc.connect("Driver={SQL Server Native Client 11.0};" "Server=(local);" "Database=Shivat_Prod;" "Trusted_Connection=yes;")
        # server = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};" "Server=(10.110.110.9);" "Database=Shivat_Prod;" "UID=sayiqan;" "PSW=Sayiqan17;" "Trusted_Connection=no;")
        # server = pyodbc.connect(
        #     "Trusted_Connection=no;DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.110.110.9;PORT=1433;DATABASE=Shivat_Prod;UID=sayiqan;PWD=Sayiqan17;")
    elif (sys.platform == "linux"):
        import pymssql
        print("running on Linux, Python version ", platform.sys.version)
        server = pymssql.connect(server="10.110.110.6", database="Stage_Lights2", port="1433", user="srulikbb",
                                 password="Srulik3141")

    mycursor = server.cursor()

    # read existing categories and their scores
    print("Reading data")
    dic_score = {}
    sql_cats = """SELECT [CategoryID]
    	  ,[CategoryName]
          ,[CategoryScore]
          FROM [Category]"""
    df_cats = pd.read_sql_query(sql_cats, server)
    for k in range(len(df_cats)):
        dic_score[df_cats['CategoryID'][k]] = df_cats['CategoryScore'][k]
    dic_score['0'] = 0

    # F = open("InternalPostID test counter.txt",'r')
    # minPostId = int(F.read())
    # F.close()
    maxPostId = minPostId = 0
    # mycursor = server.cursor()
    print(mode)
    # load selection of posts to check
    if (mode == 'latest'):
        mycursor.execute("SELECT max(InternalPostId) from Post where PostLanguage = 'ru' and AntisemiteLevel > 0")
        minPostId = maxPostId = mycursor.fetchone()[0]
        logger.info("Latest, Min post Id found is %s" % minPostId)
        if (str(minPostId) == 'None'):
            minPostId = 0
            print("starting at %s" % minPostId)
        # minPostId = 131052185
        par = {}
        par['min'] = minPostId
        query = "SELECT TOP (1000) * FROM [Stage_Lights2].[dbo].[Post]"
        df_server = pd.read_sql_query(query, server)

    if (mode == 'rerun'):
        # go from end backwards
        if (os.path.isfile("Next InternalPostID for rerun.txt")):
            F = open("Next InternalPostID for rerun.txt", 'r')
            maxPostId = int(F.read())
            F.close()

            # if (maxPostId < 950000000):
            #     print("Stopping: rerun limit reached")
            #     sys.exit()
            logger.info("In Rerun, Top post Id is %s" % maxPostId)
        else:
            mycursor.execute("SELECT max(InternalPostId) from Post where PostLanguage = 'ru'")
            maxPostId = mycursor.fetchone()[0]
            logger.info("Max post Id found is %s" % maxPostId)

        minPostId = maxPostId
        query = "SELECT top(320000) [InternalPostID],[PostText] FROM [Post] WHERE PostLanguage='ru' and InternalPostId <= " + str(
            maxPostId) + " order by InternalPostId desc"
        df_server = pd.read_sql_query(query, server)

    if (df_server.shape[0] == 0):
        print("Nothing to do")
        sys.exit()

    # print('starting')
    # print(minPostId)
    # sys.exit()a
    logger.info("--- FOCS Starting at %s seconds, %s posts to check ---" % (time.time() - start_time, df_server.shape[0]))

    return df_server, dic_score, minPostId, mycursor, server, mode, maxPostId
    print("--- FOCS Starting at %s seconds, %s posts to check ---" % (time.time() - start_time, df_server.shape[0]))


def write(test_text_InternalPostId,  test_text_cat, dic_score, mycursor, server, mode, df_server, tagger, y_probability):
    # minPostId = 0
    # cnt = cnt_cat = 0
    # ins_cat = []
    # del_list = set()
    # dic_ins_val = dict((v, set()) for v in dic_score.values())
    # for i in range(len(test_text_cat)):
    #     cnt += 1
    #     intID = str(test_text_InternalPostId[i])
    #     cat = str(test_text_cat[i])
    #
    #     del_list.add("'" + intID + "'")
    #     score = dic_score[cat]
    #     dic_ins_val[score].add("'" + intID + "'")
    #
    #     if (cat != '0'):
    #         ins_cat.append('(' + intID + ",'" + str(cat) + "')")
    #         cnt_cat += 1
    #         if (int(intID) < minPostId):
    #             minPostId = int(intID)
    #
    #     if (cnt > 999 or cnt_cat > 999):
    #         sql_del = "delete from xPost_Category where InternalPostID in (" + ','.join(del_list) + ')'
    #         mycursor.execute(sql_del)
    #         for score in dic_ins_val.keys():
    #             if (len(dic_ins_val[score]) > 0):
    #                 id_list = ','.join(dic_ins_val[score])
    #                 sql_100 = "update Post set Antisemitelevel=" + str(score) + " where InternalPostID in (" + id_list + ")"
    #                 mycursor.execute(sql_100)
    #         if (len(ins_cat) > 0):
    #             sql_ins = "insert into xPost_Category (InternalPostID, CategoryID) values " + ','.join(ins_cat)
    #             mycursor.execute(sql_ins)
    #
    #         server.commit()
    #
    #         ins_cat = []
    #         del_list = set()
    #         dic_ins_val = dict((v, set()) for v in dic_score.values())
    #         cnt = cnt_cat = 0
    #
    # if (cnt > 0):
    #     sql_del = "delete from xPost_Category where InternalPostID in (" + ','.join(del_list) + ')'
    #     mycursor.execute(sql_del)
    # for score in dic_ins_val.keys():
    #     if (len(dic_ins_val[score]) > 0):
    #         id_list = ','.join(dic_ins_val[score])
    #         sql_100 = "update Post set Antisemitelevel=" + str(score) + " where InternalPostID in (" + id_list + ")"
    #         mycursor.execute(sql_100)
    # if (len(ins_cat) > 0):
    #     sql_ins = "insert into xPost_Category (InternalPostID, CategoryID) values " + ','.join(ins_cat)
    #     mycursor.execute(sql_ins)
    #
    # server.commit()

    ####  new version    #####################################################

    PostLanguage = 'ar'
    InternalPostID_cou = 0
    cou = 0
    ins_tag = []
    print("num of persons:")
    for i in range(len(test_text_cat)):
        if(test_text_cat[i]!=0):
            InternalPostID = test_text_InternalPostId[i]
            Tagger=tagger
            TagType='Null'
            TagTypeID=test_text_cat[i]
            TagValue=y_probability[i]
            TagData='Null'
            variables = [InternalPostID ,Tagger, TagType, TagTypeID, TagValue, TagData]
            variables = [str(var) for var in variables]
            ins_tag.append("'" + "','".join(variables) +"'")
            InternalPostID_cou = InternalPostID_cou + 1
            if (len(ins_tag) > 0 and (InternalPostID_cou % 1000) == 999):
                sql_ins = "insert into PostTag (InternalPostID, Tagger, TagType, TagTypeID, TagValue, TagData) VALUES " + '(' + '),('.join(
                    ins_tag) + ')'
                sql_copy_PostTag_to_xPostCat = "insert into xPost_Category(InternalPostID, CategoryID) " \
                                               "select  InternalPostID, TagTypeID from [Stage_Lights2].[dbo].[PostTag]"
                # print(sql_ins)
                mycursor.execute(sql_ins)
                # try:
                #     mycursor.execute(sql_copy_PostTag_to_xPostCat)
                # except:
                #     print('error')
                server.commit()
                ins_tag = []
                sql_ins = []

    if (len(ins_tag) > 0):
        sql_ins = "insert into PostTag (InternalPostID, Tagger, TagType, TagTypeID, TagValue, TagData) VALUES " + '(' + '),('.join(
            ins_tag) + ')'
        mycursor.execute(sql_ins)
        server.commit()
        ins_tag = []
        sql_ins = []

    server.commit()
    mycursor.close()

df_server, dic_score, minPostId, mycursor, server, mode, maxPostId = server_connection_mode(mode='latest')


db_write(dic_score, text_predict, df_server, minPostId, mycursor, server, mode, maxPostId)