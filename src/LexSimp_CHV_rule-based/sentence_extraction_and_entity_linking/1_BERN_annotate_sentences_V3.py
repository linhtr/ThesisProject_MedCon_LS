##
import mysql.connector
import requests
import json
import time
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from nltk.tokenize.regexp import WordPunctTokenizer


##
# Input: Given DB credentials, DB Host, URL BERN and text
# Output: An extracted dataframe of the text input with BERN annotations
def BERN_annotation(args, doc_id, text, sentences_df):
    DB_HOST = args.db_host
    DB_SCHEMA = 'umls'
    DB_USER = args.db_username
    DB_PASSWORD = args.db_password
    DB_PORT = 3301
    # BERN host
    URL_BERN = args.bern_host

    # Call BERN for annotating text
    try:
        body_data = {"param": json.dumps({"text": text})}
        if URL_BERN == 'https://bern.korea.ac.kr/plain':
            # response = requests.post(url_orig, data=body_data)
            response = requests.post(URL_BERN, data={"sample_text": text}).json()
            # print("response: \n", response)
        else:
            response = requests.post(URL_BERN, data=body_data).json()
        with open(f'./output/subtitles/BERN_annotated_subtitles_{doc_id}.txt', 'w') as outfile:
            json.dump(response, outfile)
    except Exception as e:
        print(e)
        print("Error when calling BERN ")

    # parse BERN annotations from JSON
    try:
        count = 0
        _ = ''
        extracted_df = []

        for el in response['denotations']:
            CUIS = []
            MESH_code = 'Unknown'
            for code in el['id']:
                if 'MESH' in code:
                    MESH_code = code.replace('MESH:', '')
                    query_MESHtoCUI = "SELECT distinct CUI FROM umls.MRCONSO WHERE CODE=%(MESH_CODE)s and SAB='MSH';"
                    try:
                        # Connect to research DB
                        try:
                            connection = mysql.connector.connect(host=DB_HOST,
                                                                 database=DB_SCHEMA,
                                                                 user=DB_USER,
                                                                 password=DB_PASSWORD,
                                                                 port=DB_PORT,
                                                                 use_pure=True)
                            cursor = connection.cursor()
                            print('successfully connected to MySQL')
                        except mysql.connector.Error as e:
                            print("Error while connecting to MySQL", e)

                        cursor.execute(query_MESHtoCUI, params={'MESH_CODE': MESH_code}, multi=False)
                        result = cursor.fetchall()
                        for tuple in result:
                            CUIS.append(tuple[0])

                        # Close connection to research DB
                        if connection.is_connected():
                            connection.close()
                            print('-------------------------------------------------')
                            print("MySQL connection is closed\n")

                    except:
                        print('Unable to retrieve CUI for MESH code: ' + MESH_code)

            # extract the annotated sentence from the entire document
            start_span = el['span']['begin']
            end_span = el['span']['end']
            sentence_begin = text.rindex("\n", 0, start_span)
            sentence_end = text.index("\n", end_span)
            # sentence = str(text[sentence_begin+3:sentence_end]) # when using "***" as separator
            sentence = str(text[sentence_begin+1:sentence_end])  # when using "\n" as separator
            print(sentence)

            sent_id = sentences_df.index[sentences_df == sentence].tolist()[0]
            print("sent_id: ", sent_id)

            # count entities per sentence
            if sentence != _:
                count = 1
                _ = sentence
            else:
                count += 1

            entity_count = count
            entity_type = str(el['obj'])
            extracted_ngram = str(text[start_span:end_span])

            print(
                f'Entity {entity_count}: {extracted_ngram} \nEntity type: {entity_type} \nMESH ID: {MESH_code} \nCUIs: {CUIS} \n')

            extracted_df.append(OrderedDict({"sent_id": sent_id,
                                             "sentence": sentence,
                                             "entity_count": entity_count,
                                             "BERN_extracted_ngram": extracted_ngram,
                                             "BERN_entity_type": entity_type,
                                             "msh_id": MESH_code,
                                             "MESHtoCUIs": CUIS
                                             }))

        extracted_df = pd.DataFrame(extracted_df,
                                    columns=["sent_id", "sentence", "entity_count", "BERN_extracted_ngram",
                                             "BERN_entity_type", "msh_id", "MESHtoCUIs"])

    except Exception as e:
        print(e)
        print('Error while parsing the BERN annotations...')

    return extracted_df


##
if (__name__ == '__main__'):
    # PARAMETERS
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--db_username', default='abc')
    argparser.add_argument('--db_password', default='123')
    argparser.add_argument('--db_host', default="myt-research-cluster.cluster-cqjr4dunkiqp.eu-west-1.rds.amazonaws.com")
    argparser.add_argument('--bern_host', default="https://bern.korea.ac.kr/plain")
    # argparser.add_argument('--bern_host', default='http://34.243.241.63:8888/') # myT BERN host

    params = argparser.parse_args()
    ##
    project_path = '/Users/linh/Documents/GitHub/Thesis_Project/'

    # Import data
    data = pd.read_csv(project_path + 'datasources/opensubtitles/clean_house_subs.csv')
    print('Number of sentences: ', len(data))
    print(data.head())
    print(data.tail())

    sentences = data["sentence"]
    print(f'\nsentences:\n{sentences.head()}\n')

    # # Feed each sentence separately to BERN (takes a long time)
    # for i,sentence in enumerate(tqdm(sentences)):
    #     print(i, sentence)
    #     sentence_id = i
    #     BERN_annotated_df = BERN_annotation(params, sentence, sentence_id)
    #     # If csv doesn't exist yet, create one, and append to it in the next iterations
    #     with open('BERN_annotated_subtitles.csv', 'a') as f:
    #         BERN_annotated_df.to_csv(f, header=f.tell()==0, index=False, sep=';')

    # To speed up the BERN annotation process, the sentences are concatenated in multiple large text documents
    document='\n'
    documents = []
    for i, sentence in enumerate(sentences, 1):
        document += sentence + '\n'
        # Adjust number to make sure that a document doesn't exceed 15000 tokens
        if i % 500 == 0:
            documents.append(document)
            document = '\n'
    documents.append(document)

    print(f'Amount documents: {len(documents)}\n')

    # Check word count for every document
    # for i,doc in enumerate(documents):
    #     word_count = len(doc.split())
    #     print(f'Doc {i}, word_count: {ord_count}')

    # Check token count for every document (should not be more than 15000)
    for i, doc in enumerate(documents):
        token_count = WordPunctTokenizer().tokenize(doc)
        # print(token_count)
        print(f'Doc {i}, token count: {len(token_count)}')
    ##
    # Feed each text document separately to BERN
    for i, doc in enumerate(tqdm(documents), 14):
        BERN_annotated_df = BERN_annotation(params, i, doc, sentences)
        # If csv doesn't exist yet, create one, and append to it in the next iterations
        with open('./output/subtitles/BERN_annotated_subtitles.csv', 'a') as f:
            BERN_annotated_df.to_csv(f, header=f.tell() == 0, index=False, sep=';')
        time.sleep(5)
