
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from transformers import DistilBertTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from constants import *
from keybert import KeyBERT
import pandas as pd
import re
import json
from langchain_chroma import Chroma



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_overlap=50,chunk_size=600)
kw_model = KeyBERT()


         #'globalpartnerships':['capacity-development','finance','multi-stakeholder-partnerships','science','trade','national-sustainable-development-strategies']}

def preprocess(text):
    text=re.sub(r'<[^>]*>','',text)
    text=re.sub(r"[^A-Za-z" "]+"," ",text).lower()
    #text=re.sub("[0-9" "]+"," ",text)
    text=re.sub(r'[\W]+',' ',text.lower()).replace('-','')
    if re.findall(text,r'what can i do'):
        text=""
    return text
def webprofiler(base='https://www.un.org/sustainabledevelopment/',subpage='',\
                GoalNumber=1,GoalLabel=""):
    full_path=base+subpage
    loader=WebBaseLoader(web_paths=[full_path])
    docs=loader.load()
    doc_chunks=splitter.split_documents(docs)
    corpus=[]
    keyword_list=[]
    for d in doc_chunks:
        if "HomeOverview" in d.page_content:continue
        if len(d.page_content)<800:continue #Skips to body of text
        #keywords = kw_model.extract_keywords(d.page_content,keyphrase_ngram_range=(1, 2))
        #for k in keywords:
        #    if k[1]<0.45:continue
        #    keyword_list.append(k[0])
        sents=d.page_content.split('\n')
        if sents[-1]=="Links":break #End of page
        for s in sents:
            single_sentances=s.split('.')
            for ss in single_sentances:
                if 'please visit this link' in ss:continue
                if len(ss.split())<7:continue #ignore sentence fragments with few words
                #print(ss, len(ss))
                #if len(ss)>14:
                corpus.append(ss.lower()) 
    for t in corpus:
        keywords = kw_model.extract_keywords(t,keyphrase_ngram_range=(1, 2))
        keyword_list.append(keywords[0][0])        
    df=pd.DataFrame(data={'text_data':corpus,'source':full_path})
    df['text_data'] = df.text_data.apply(preprocess)
    df['best_key_word']=keyword_list
    mask=df['text_data'].str.len()>14 #ignore fragments with very few char
    df_filtered=df[mask]
    text=df_filtered.text_data.to_list()
    df_filtered['category']='Goal %d: %s' %(GoalNumber,GoalLabel)
    df_filtered['label']=GoalNumber
    keyword_list=set(keyword_list)
    return df_filtered,keyword_list

if __name__ == '__main__':
    goal_count=1
    keyword_dict={}
    for i,v in sdg_goals.items():
        #if i!='globalpartnerships':continue
        print("source: https://www.un.org/sustainabledevelopment/%s" %i) 
        print("Goal %d: %s" %(goal_count,v))
        df_un,un_keyword=webprofiler(subpage=i,GoalNumber=goal_count,GoalLabel=v)
        features_list=[df_un]
        for subpage in sdg_topics[i]:
            df_unsdg,new_keywords=webprofiler(base='https://sdgs.un.org/topics/',subpage=subpage,GoalNumber=goal_count,GoalLabel=v)
            un_keyword=un_keyword.union(new_keywords)
            features_list.append(df_unsdg)
        df_all=pd.concat(features_list)
        keyword_dict[i]=list(un_keyword)
        df_all.to_csv("training_data/FeatureSet_%d.csv" %goal_count)
        #print(df_all.head())
        goal_count+=1
        #break;
    with open('training_data/KeywordPayload.json','w') as fp:
        json.dump(keyword_dict, fp)
