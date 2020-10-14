from os import listdir
from os.path import isfile, join
import xml.etree.cElementTree as ET

import logging, sys
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)s)"
LOG_DATE_FORMAT='%Y-%m-%d %H:%M:%S'

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

namespaces = {'graph': "http://www.xces.org/ns/GrAF/1.0/",
              "xmlns:graf": "http://www.xces.org/ns/GrAF/1.0/",
              'xml': "http://www.w3.org/XML/1998/namespace"}

def do_merge(path):
    inDir = path

    out = open(path + '/output.txt', 'w')

    seg_xml_files = [f for f in listdir(inDir) if isfile(join(inDir, f)) and f.endswith("seg.xml")]
    for seg_xml_file in sorted(seg_xml_files):

        seg_xml_file = join(inDir, seg_xml_file)
        logger.info(seg_xml_file)
        regions=dict()
        seg_root = ET.parse(seg_xml_file).getroot()
        for region_ele in seg_root.findall("graph:region", namespaces):
            xml_id = region_ele.attrib['{http://www.w3.org/XML/1998/namespace}id']
            regions[xml_id] = region_ele.attrib['anchors']
            # logger.info("Region: " + xml_id + " -> " + regions[xml_id])

        sent_xml_file = seg_xml_file.replace('-seg.xml', '-sent.xml')
        sents=dict()
        sent_root = ET.parse(sent_xml_file).getroot()
        for a_ele in sent_root.findall("graph:a", namespaces):
            sent=dict()
            if not a_ele.attrib['label'] =='s':
                continue
            ref = a_ele.attrib["ref"]
            graph_node = sent_root.find('.//graph:node[@xml:id="' + ref + '"]', namespaces)
            for link in graph_node:
                target_regions = link.attrib['targets']
                sent["sent_start"] = int(str.split(regions[target_regions])[0])
                sent["sent_end"] = int(str.split(regions[target_regions])[1])
                logger.debug("Sentence: " + ref + " -> " + regions[target_regions] )

            sent['tokens'] = dict()
            sents[sent['sent_start']] = sent


        pos_xml_file = seg_xml_file.replace('-seg.xml', '-pos.xml')
        pos_root = ET.parse(pos_xml_file).getroot()
        for a_ele in pos_root.findall("graph:a", namespaces):
            tok=dict()
            ref = a_ele.attrib["ref"]
            tok['id'] = ref
            graph_node = pos_root.find('.//graph:node[@xml:id="' + ref + '"]', namespaces)
            my_sent = None
            for link in graph_node:
                target_regions = link.attrib['targets']
                logger.debug("Token: " + ref + " -> " )
                tok_start = int(str.split(regions[target_regions])[0])
                tok_end = int(str.split(regions[target_regions])[1])
                tok['start'] = tok_start
                tok['end'] = tok_end
                tok['regions'] = target_regions
            for fs_ele in a_ele:
                for f_ele in fs_ele:
                    tok[f_ele.attrib['name']] = f_ele.attrib['value']
            for sent_id in sents:
                if tok_start >= sents[sent_id]['sent_start'] and  tok_end <= sents[sent_id]['sent_end']:
                    my_sent = sents[sent_id]
                    break
            if my_sent:
                sentOrd = tok['sentOrd']
                my_sent['tokens'][sentOrd] = tok
            else:
                pass
                logger.warning("Nosent?: " + tok)


        for sent_id, sent in sorted(sents.items(),  key=lambda item: int(item[0])):
            #print("Sentence offsets: " + str(sent['sent_start']) + '-' + str(sent['sent_end']))
            out.write("Sentence offsets: " + str(sent['sent_start']) + '-' + str(sent['sent_end']) + '\n')
            for tokenSentOrd, token in sorted(sent['tokens'].items(), key=lambda item: int(item[0])):
                #print("Token: " + str(token))
                out.write("Token: " + str(token)+'\n')
            print("")

        ners=dict()
        ner_xml_file = seg_xml_file.replace('-seg.xml', '-ner.xml')
        ner_root = ET.parse(ner_xml_file).getroot()
        for a_ele in ner_root.findall("graph:a", namespaces):
            ner = dict()
            ref = a_ele.attrib["ref"]
            graph_node = ner_root.find('.//graph:node[@xml:id="' + ref + '"]', namespaces)
            for link in graph_node:
                target_regions = link.attrib['targets']

                ner['id'] = ref
                ner_start = int(str.split(regions[target_regions])[0])
                ner_end = int(str.split(regions[target_regions])[1])
                ner['start'] = ner_start
                ner['end'] = ner_end
                ner['regions'] = target_regions
                ner['label'] =  a_ele.attrib["label"]
                for fs_ele in a_ele:
                    for f_ele in fs_ele:
                        ner[f_ele.attrib['name']] = f_ele.attrib['value']

                ners[ner['id']] = ner

        for ner_id, ner  in ners.items():
            #print("Ner: " + str(ners[ner_id]))
            out.write("Ner: " + str(ners[ner_id])+'\n')

        print()

if __name__ == '__main__':
    do_merge('xmls_tree/9')