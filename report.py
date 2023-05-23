import datetime

class Report(object):
    def __init__(self):
        self.time = datetime.datetime.now()
        self.remarks = list()
        self.lines = list()
        self.header = """
        #  MCMC analysis of individual components
        #========================================
        %s
        """ % self.time.strftime("%Y %b %d  %H:%M:%S")
        self.report = ""

    def add_line(self, text):
        if not text.endswith('\n'):
            text += '\n'
        self.lines.append(text)

    def add_linebreak(self):
        self.lines.append("\n")

    def add_remark(self, text):
        self.remarks.append(text)

    def make_report(self):
        remark_str = ''.join(self.remarks)
        lines_str = ''.join(self.lines)
        self.report = '\n'.join([self.header, remark_str, lines_str])

    def print_report(self):
        self.make_report()
        print(self.report)

    def write(self, fname):
        self.make_report()
        with open(fname, 'w') as output:
            output.write(self.report)

