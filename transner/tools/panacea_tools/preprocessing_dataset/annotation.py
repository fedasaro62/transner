import re
import ast
import os 

def annotate(path):
    data = open(path + '/output.txt', 'r').readlines()

    out = open('conll_final/'+ os.path.splitext(os.path.basename(path))[0] + '.conll', 'w')
    #print('Created file ' + out.name)

    ners = []

    #for matches in re.findall(r"Ner.*", data):
    for line in data:
        if line.startswith('Ner:'):
            ner = ast.literal_eval(line[5:len(line)-1])

            if float(ner['conf']) >= 0.75:
                ners.append(ner)


    ners.sort(key=lambda  x: x['start'])

    end_offset = -1

    for line in data:
        if line.startswith('Sentence'):
            end_offset = int(re.sub('Sentence offsets: ', '', line).split('-')[1])
        if line.startswith('Token:'):
            found = False
            tok = ast.literal_eval(line[7:len(line)-1])
            start, end = tok['start'], tok['end']
            for ner in ners:

                if start >= ner['start'] and end <= ner['end']:
                    out.write(tok['word'] + '\t' + ner['label']+'\n')
                    #print(tok['word'] + '\t' + ner['label'])
                    found = True

                    if end == end_offset:
                        out.write('\n')
                    continue
            
            if not found:
                out.write(tok['word'] + '\tO\n')
                if end == end_offset:
                    out.write('\n')
                #print(tok['word'] + '\tO')

if __name__ == "__main__":
    for subdir, dirs, files in os.walk('xmls_tree'):
        for dir in dirs:

            #print('Parsing xml in ' + subdir + '/' + dir)
            try:
                annotate(subdir+'/'+dir)
            except OSError as e:
                print(str(e))
                continue

    if os.path.exists('panacea_dataset.conll'): os.remove('panacea_dataset.conll') 
    out = open('panacea_dataset.conll', 'a')
    for file in os.listdir('conll_final/'):
        data = open('conll_final/' + file, 'r').read()
        out.write(data)


