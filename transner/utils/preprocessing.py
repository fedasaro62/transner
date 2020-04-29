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


class NERSeparatePunctuations(NERPreprocessing):
    """This class contains the splitting for composite punctuation words like 'dell'Italia', 'L'America'.
    """
    def __init__(self):
        super(NERSeparatePunctuations, self).__init__()
        self.puncts = string.punctuation


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
            curr_changes = []
            add_ws = False
            for offset, ch in enumerate(s):
                #pdb.set_trace()
                if add_ws:
                    if ch != ' ':
                        proc_string += ' '
                        curr_changes.append(len(proc_string)-1)
                    add_ws = False
                if ch in self.puncts:
                    if offset > 0 and proc_string[-1] != ' ':
                        proc_string += ' '
                        curr_changes.append(len(proc_string)-1)
                    add_ws = True
                proc_string += ch
            if do_lower_case:
                proc_string = proc_string.lower()
            proc_strings.append(proc_string)
            self.changes.append(curr_changes)
        return proc_strings


    #TODO solve bug
    def adjustEntitiesOffset(self, entities, adjust_case=False):
        """Readjust the entities offset due to preprocessing of the original strings

        Arguments:
            entities {list} -- List of entities, ordered by the original strings indexes, for each string having the format [{["offset": ]}]
            adjust_case {boolean} -- if True, the value of each entity is copied from the span in the original string (to avoid case mismatch)
        """

        assert len(entities) == len(self.original)
        for s, e_list, c_list in zip(self.original, entities, self.changes):
            for e in e_list:
                curr_count = 0
                for c in c_list:
                    #pdb.set_trace()
                    if e['offset'] >= c:
                        curr_count += 1
                    else:
                        break
                e['offset'] -= curr_count

        if adjust_case:
            # copy the entities values from the original string
            for s, e_list in zip(self.original, entities):
                for e in e_list:
                    start_pos = e['offset']
                    end_pos = start_pos + len(e['value'])
                    e['value'] = s[start_pos:end_pos]
                    # clean possible white spaces (unlikely to happen)
                    if e['value'][-1] == ' ':
                        e['value'] = e['value'][:-1]
