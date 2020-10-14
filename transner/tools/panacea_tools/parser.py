import os 
import xml.etree.ElementTree as ET
from lxml import etree
import bs4

import re

def retrieve_entities() :

    for file in os.listdir('xmls/'):
        file_name, file_extension = os.path.splitext(os.path.basename(file))
        if file.endswith('ner.xml'):
            entities = {}
            with open('xmls/'+file, 'r') as f:
                for line in f:          
                    if line.strip().startswith('<!--'):
                        start = re.sub('<!--', '', line)
                        end = re.sub('-->', '', start)
                        key = end.strip()
                    elif line.strip().startswith('<a label='):
                        start = re.sub('<a label=\"', '', line)
                        end = re.sub('\" ref.*', '', start)
                        value = end.strip()
                    elif line.strip().startswith('<f '):
                        start = re.sub('<f name=\"conf\" value=\"', '', line)
                        end = re.sub('\"/>', '', start)
                        acc = float(end.strip())
                        if key not in entities and acc >= 0.85:
                            entities.update({key: value})
            
            #with open('xmls/'+file_name.split('-')[0]+'-plain.txt') as ft:
            annotation('xmls/'+file_name.split('-')[0]+'-plain.txt', entities)
            
    return entities 

def annotation(file, entities):
    file_name, file_extension = os.path.splitext(os.path.basename(file))   
    
    with open(file) as f:
        out = open('out/'+file_name+"_annotated"+file_extension, 'w') #file annotated

        data = f.read()
        for key in entities.keys():
            new_line = ' $' + key + '$' + entities[key] + ' '
            if str.find(data, ' '+key+' ') > 1:
                #print(key, entities[key])
                data = data.replace(' '+key+' ', new_line)
                
                       
        out.write(data)
    out.close()
    create_conll('out/'+file_name+"_annotated"+file_extension)

def create_conll(path):
    file_name, file_extension = os.path.splitext(os.path.basename(path))
    out = open('conll/' + file_name + '.conll', 'w')
    with open(path) as f:
        for line in f:
            entity = False
            annotation = []
            for word in line.split():
                #print(word)
                if word.startswith('$') or entity is True:
                    if word.count('$') == 2:
                        #print('single entity')
                        out.write(word.split('$')[1] + ' ' + word.split('$')[2] + '\n')

                    elif word.count('$') == 1 and entity is True:
                        #print('end of entity')
                        tag = word.split('$')[1]
                        annotation.append(word.split('$')[0])
                        for w in annotation:
                            out.write(w + ' ' + tag + '\n')
                        annotation = []
                        entity = False
                    
                    elif word.count('$') == 1 and entity is False:
                        #print('start of entity')
                        entity = True
                        annotation.append(word.split('$')[1])
                        
                    else:
                        #print('middle entity')
                        annotation.append(word)
                else: 
                    out.write(word + ' O\n')
                #input()


def retrieve_entities_from_file():
    f = open('entities.txt', 'r')
    entities = {}
    for line in f:
        entity = line.rstrip('\n')
        flag = entity.split('$')
        k, v = flag[0], flag[1]
        #FIXME: some duplicated keys with different values are not imported
        entities.update({k: v}) 
    return entities

def clean_entities(entities):
    delete_keys = []
    tmp = entities.copy()
    for key in entities.keys():
        #print('orginal_dict:' + key)
        tmp.pop(key)
        #print('deleted ' + kX +' '+vX)
        for k in tmp.keys():
            #print('tmp dict:'+k)
            if (re.search(r'\b'+key+r'\b', k)) is not None:
                #print('\t key found :'  + k)
                delete_keys.append(key)

    for key in delete_keys:
        if key in entities.keys():
            entities.pop(key)

    return entities


#FIX ME: the dataset in coll is 22B length. balance function need to be implemented
if __name__ == "__main__":
    entities = retrieve_entities()

    for file in os.listdir('conll/'):
        output_file = open('output_greek.conll', 'a')
        data = open('conll/' + file, 'r').read()
        data = data + '\n'
        output_file.write(data)