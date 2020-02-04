import subprocess




def class NER_interface():

    @staticmethod
    def set_taxonomy(taxonomy):
        pass

    @staticmethod
    def contact_NER(input_str):
        args = ("bin/bar", "-c", "somefile.xml", "-d", "text.txt", "-r", "aString", "-f", "anotherString")
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        return output


