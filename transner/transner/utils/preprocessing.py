from abc import ABC, abstractmethod
import pdb
import string


class NERPreprocessing(ABC):
    """Abstract class for the preprocessing of NER input
    """
    @abstractmethod
    def __init__(self):
        self.changes = []
        self.original = []
        super(NERPreprocessing, self).__init__()

    @abstractmethod
    def preprocess(self, strings):
        """Preprocessing of the strings

        Arguments:
            strings {String} -- Strings to preprocess
        """
        pass

    @abstractmethod
    def adjustEntitiesOffset(self, entities):
        """Readjust the entities offset due to preprocessing of the original strings

        Arguments:
            entities {list} -- List of entities for each string having the format [{["value": val, "type": type, "offset": ]}]
        """
        pass


class NERSeparatePunctuations():
    """This class contains the splitting for composite punctuation words like 'dell'Italia', 'L'America'.
    """
    def __init__(self):
        super(NERSeparatePunctuations, self).__init__()
        self.reset()

    def reset(self):
        self.puncts = string.punctuation + "“" + "”"
        self.proc2origin = [] #maps offset_processed -> offset_original
        self.original = []


    def preprocess(self, strings, do_lower_case=False):
        """Insert whitespaces around punctuations to separate them from words. Ex. "L'America" becomes "L ' America".

        Arguments:
            strings {list} -- List of strings to preprocess

        Returns:
            [list] -- List of preprocessed strings
        """
        self.original = strings
        proc_strings = []
        for s in strings:
            proc_string = ''
            offset_mapping = []
            for original_offset, ch in enumerate(s):
                if ch == ' ' and s[original_offset-1] in self.puncts:
                    #if this is a space after a punctation just do nothing
                    # because the space was already added to the processed string
                    pass
                elif ch in self.puncts:
                    #surrounding punctuation with white spaces
                    # if the first white space is already present then do nothing otherwise insert it manually
                    if original_offset > 0 and s[original_offset-1] != ' ' and proc_string[-1] != ' ':
                        proc_string += ' '
                        offset_mapping.append(-1)
                    proc_string += ch
                    offset_mapping.append(original_offset) #punctuation mapping
                    if len(s) > original_offset+1:
                        #if the punctuation is not the final character
                        proc_string += ' '
                        #if the original string already has a space after the punctuation, just map the space to the original one
                        # else map to -1: character not present in the original string
                        offset_mapping.append(original_offset+1 if s[original_offset+1] == ' ' else -1)
                else:
                    proc_string += ch
                    offset_mapping.append(original_offset)

            if do_lower_case:
                proc_string = proc_string.lower()
            assert len(proc_string) == len(offset_mapping), 'processed string and offset mapping lengths do not match'
            proc_strings.append(proc_string)
            self.proc2origin.append(offset_mapping)
        return proc_strings


    def adjustEntitiesOffset(self, entities, adjust_case=False):
        """Readjust the entities offset due to preprocessing of the original strings

        Arguments:
            entities {list} -- List of entities, ordered by the original strings indexes for each string having the format [{["offset": ]}]
            adjust_case {boolean} -- if True, the value of each entity is copied from the span in the original string (to avoid case mismatch)
        """
        assert len(entities) == len(self.original), 'list of entities length do not match the number of original strings'
        for e_list, offset_mapping in zip(entities, self.proc2origin):
            #adjust offsets
            for e in e_list:
                e['offset'] = offset_mapping[e['offset']]

        if adjust_case:
            # copy the entities values from the original string to resume the case information
            for s, e_list, offset_mapping in zip(self.original, entities, self.proc2origin):
                for e in e_list:
                    start_pos = e['offset']
                    non_existing_tokens = 0
                    #compute number of non-existing tokens in the original string
                    for i in range(start_pos, start_pos+len(e['value'])):
                        non_existing_tokens += int(offset_mapping[i] == -1)

                    end_pos = start_pos + len(e['value']) - non_existing_tokens
                    e['value'] = s[start_pos:end_pos]
                    # clean possible white spaces (unlikely to happen)
                    if e['value'][-1] == ' ':
                        e['value'] = e['value'][:-1]
