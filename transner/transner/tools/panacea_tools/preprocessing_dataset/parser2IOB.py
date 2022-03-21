# Using readlines() 
file = open('panacea_original/panacea_dataset.conll', 'r', encoding='utf-8') 
lines = file.readlines() 

out = open('panacea_original/panacea_dataset_iob.conll', 'w')
  
count = 0
for line in lines: 
	if line != '\n':
		tok = line.strip().split('\t')
		
		tokT = 'O'

		if(tok[1] == 'ORG') :
			tokT = 'I-ORG'

		if(tok[1] == 'PER') :
			tokT = 'I-PER'

		if(tok[1] == 'LOC') :
			tokT = 'I-LOC'

		out.write(tok[0] + '\t' + tokT+'\n')
	else:
		out.write('\n')