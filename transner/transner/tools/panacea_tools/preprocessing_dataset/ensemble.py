import annotation
import graf_merger

import os
from os import path

if __name__ == "__main__":
    #for file in os.listdir('xmls'):
    #    file_name, file_extension = os.path.splitext(os.path.basename(file))
        #print(file)
    #    file_index = file_name.split('-')[0]

    #    if not os.path.exists('xmls_tree/' + file_index):
    #        os.mkdir('xmls_tree/' + file_index)
    #    os.rename("xmls/" + file_name + file_extension, 'xmls_tree/' + file_index + '/' + file_name + file_extension)

    out = open('log.txt', 'w')
    for subdir, dirs, files in os.walk('xmls_tree'):
        if len(files) == 7:
            if not path.exists(subdir+'/output.txt'):
                try:
                    graf_merger.do_merge(subdir)
                except OSError as e:
                    out.write(str(e))
                    out.write('\n')
                    continue
