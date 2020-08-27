from treetagger import treetaggerwrapper as ttw
import pandas as pd
from tqdm import tqdm
import mysql.connector
import requests
import json
import time
import argparse
import logging
import sys

# The Database class comes from Manolis.
# The rest of the script is adapted from the TreetaggerAnnotation script from Erick.

class Database:
    def __init__(self,DB_HOST,DB_SCHEMA,DB_USER,DB_PASSWORD,DB_PORT):
        try:
            self.connection = mysql.connector.connect(host=DB_HOST,
                                                 database=DB_SCHEMA,
                                                 user=DB_USER,
                                                 password=DB_PASSWORD,
                                                 port=DB_PORT)
            self.cursor = self.connection.cursor()
            logging.info('successfully connected to MySQL')
        except mysql.connector.Error as e: \
            logging.exception("Error while connecting to MySQL", e)
    @property
    def get_connection(self):
        return self.connection

    @property
    def get_cursor(self):
        return self._cursor

    def fetch_query(self,query):
        try:
            self.cursor.execute(query, multi=False)
            result = self.cursor.fetchall()
        except Exception as e:
            logging.exception('Query to DB failed: ',e)
        return result

    def run_query(self,query):
        try:
            self.cursor.execute(query, multi=False)

        except Exception as e:
            logging.exception('Query to DB failed: ',e)

    def run_multiple_queries(self,query, data):
        try:
            self.cursor.executemany(query,data)
            self.commit()
            print (self.cursor.rowcount,'was inserted.' )

        except Exception as e:
            logging.exception('Query to DB failed: ',e)

    def commit(self):
        try:
            self.connection.commit()
        except Exception as e:
            logging.exception('Commit to db failed: ',e)

    def __exit__(self):
        if self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection is closed")

def tagText(stringText):
    tags = tagger.tag_text(stringText)
    return tags

def writeFileLineByLine(stringLines, nameFile):
    outF = open(nameFile+".txt", "w")
    for l in stringLines:
        print(l, file=outF)
    outF.close()
    return

def addRegister(data_to_add):
    #print(data_to_add[0][0])
    #print(type(title_tags))
#     for d in data_to_add:
#         print('\n'.join(d[0]))
#         t_string = '\n'.join(d[0])
#         a_string = '\n'.join(d[1])
    #db.run_query('USE pubmed.medline_citation;')
    qry = """
    INSERT INTO intern_projects.wikip_med_conditions_annotations (title_annotated, text_annotated) VALUES(%s,%s)
    """
    db.run_multiple_queries(qry, data_to_add)
    #print([(title_tags),(abstract_tags),(id_s)])
    return
#.join(sentence)

def tagQueries(listOfQueries):
    ids = listOfQueries['pubId'].to_list()
    titles = listOfQueries['Title'].to_list()
    abstracts = listOfQueries['Abstract'].to_list()
    data_to_add = []
    i = 0
    while i < len(ids):
        s_id = ids[i]
        title = titles[i]
        tags_title = tagText(title)
        #writeFileLineByLine(tags_title,'title_'+str(s_id))
        abst = abstracts[i]
        tags_abstract = tagText(abst)
        #writeFileLineByLine(tags_abstract,'abstract_'+str(s_id))
        data_to_add.append(['\n'.join(tags_title),'\n'.join(tags_abstract),str(s_id)])
        i = i+1
    addRegister(data_to_add)
    return

def tag_wiki_articles(listOfArticles):
    titles = listOfArticles['title'].to_list()
    docs = listOfArticles['parsed_text'].to_list()
    data_to_add = []
    i = 0
    while i < len(listOfArticles):
        tags_title = tagText(titles[i])
        # writeFileLineByLine(tags_title,'title_'+str(i))
        tags_doc = tagText(docs[i])
        #writeFileLineByLine(tags_abstract,'abstract_'+str(i))
        data_to_add.append(['\n'.join(tags_title),'\n'.join(tags_doc)])
        i = i+1
    # addRegister(data_to_add)
    return pd.DataFrame.from_records(data_to_add, columns=['title_annotated','text_annotated'])

def main():

    df_med_conditions = pd.read_json(
        '../datasources/wikipedia/extracted/articles/medical_conditions_V4.json', lines=True, chunksize=1000)

    # i = 0
    # increment = 1000
    # pointer = 0
    #
    # #limit = 1
    # limit = 600
    # while i < limit:
    #     pointer = i*increment
    #     # trial_id_tuples=db.fetch_query('SELECT pmid, article_title, abstract_text FROM pubmed.medline_citation where abstract_text is not null LIMIT '+str(pointer)+', '+str(increment)+';')
    #     # print(pointer, increment,'{0}'.format(i/limit*100))
    #     my_pandas_db = pd.DataFrame(trial_id_tuples, columns=['pubId','Title','Abstract'])
    #     tagQueries(my_pandas_db)
    #     #sys.stdout.write('{0}'.format(i/limit*100))
    #     #sys.stdout.flush()
    #     i = i+1

    # Write to SQL DB
    # for chunk in tqdm(df_med_conditions):
    #     tag_wiki_articles(chunk.head(1))

    # Write to a local file
    for chunk in tqdm(df_med_conditions):
        tagged_articles = tag_wiki_articles(chunk)
        # If csv doesn't exist yet, create one, and append to it in the next iterations
        with open('Wiki_MedCon_TreetaggerAnnotation.csv', 'a') as f:
            tagged_articles.to_csv(f, header=f.tell()==0, index=False, sep=';')


if __name__ == '__main__':
    ############ PARAMETERS #################
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--db_username', default='abc')
    argparser.add_argument('--db_password', default='123')
    argparser.add_argument('--db_host', default="myt-research-cluster.cluster-cqjr4dunkiqp.eu-west-1.rds.amazonaws.com")

    args = argparser.parse_args()

    DB_HOST = args.db_host
    DB_SCHEMA = 'intern_projects'
    DB_USER = args.db_username
    DB_PASSWORD = args.db_password
    DB_PORT = 3301

    db = Database(DB_HOST,DB_SCHEMA,DB_USER,DB_PASSWORD,DB_PORT)

    tagger = ttw.TreeTagger(TAGLANG='en')

    main()
