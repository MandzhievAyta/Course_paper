import codecs
import ntpath
import re
from nltk import tokenize
from nltk.parse import stanford
import os
from nltk.tag import StanfordPOSTagger
from nltk.tree import ParentedTree
from IPython.display import Image, display
import datetime
import sys

def RemoveShortStubs(paragraphs):
    startStub = [u'it', u'there']
    endStub = u'that'
    for par in paragraphs:
        for sent in par:
            startidx = []
            endidx = []
            flag = 0
            for i, word in enumerate(sent):
                if word[0].lower() in startStub:
                    flag = 1
                    start_candidate = i
                if (word[0] == endStub) and (flag == 1):
                    flag = 0
                    startidx.append(start_candidate)
                    endidx.append(i)
            for stubidx in reversed(range(len(startidx))):
                del sent[startidx[stubidx]: endidx[stubidx] + 1]

def PreProcess(text):
    #Deleting literature references
    #[([] -open bracket ( or [;  [^([]*? - lazy; \d+ - number; [^])]*?; [])] - close bracket ) or ]
    #print "regex:"
    #time_beg = datetime.datetime.now()
    pat = re.compile(r"[([][^([]*?\d+[^])]*?[])]", re.IGNORECASE or re.DOTALL)
    text = re.sub(pat, r"", text)
    text = re.sub(r'([,.:;/)("])', r' \g<1> ', text)
    #print (datetime.datetime.now() - time_beg).total_seconds()
    
    #Splitting into paragraphs
    paragraphs = re.split(u"\n", text)
    
    #Splitting into sentences
    for i, par in enumerate(paragraphs):
        paragraphs[i] = tokenize.sent_tokenize(par)
    paragraphs_for_tag = []
    for par in paragraphs:
        paragraphs_for_tag.extend(par)
        
    #Tagging all words
    #print "downloading stanford pos tagger:"
    #time_beg = datetime.datetime.now()
    st = StanfordPOSTagger('english-left3words-distsim.tagger')
    #print (datetime.datetime.now() - time_beg).total_seconds()
    #time_beg_tagging = datetime.datetime.now()

    for i, par in enumerate(paragraphs_for_tag):
        paragraphs_for_tag[i] = paragraphs_for_tag[i].split()
    #print (datetime.datetime.now() - time_beg_tagging).total_seconds()
    paragraphs_for_tag = st.tag_sents(paragraphs_for_tag)
    prev_par = 0
    for i, par in enumerate(paragraphs):
        paragraphs[i] = paragraphs_for_tag[prev_par:prev_par + len(paragraphs[i])]
        prev_par += len(paragraphs[i])
    for i, par in enumerate(paragraphs):
        for sentence in range(len(par)):
            paragraphs[i][sentence][:] = [x for x in paragraphs[i][sentence] if x[0] != u'(' and x[0] != u')']
    #print "actual tagging:"
    #print (datetime.datetime.now() - time_beg_tagging).total_seconds()
    #print paragraphs
    #print "remove short stubs:"
    #time_beg = datetime.datetime.now()
    RemoveShortStubs(paragraphs)
    #print (datetime.datetime.now() - time_beg).total_seconds()
    return paragraphs

class SourceText:
    def __init__(self, paragraphs):
        self._paragraphs = [x for x in paragraphs if x != []]
        self._par_iter = 0
        self._sent_iter = 0
        
    def nextPar(self):
        self._par_iter += 1
        if self._par_iter == len(self._paragraphs):
            return 1
        else:
            self._sent_iter = 0
            return 0
        
    def nextSent(self):
        if self._sent_iter < len(self._paragraphs[self._par_iter]):
            self._sent_iter += 1
            return self._paragraphs[self._par_iter][self._sent_iter - 1], 0
        else:
            return [], 1

class OffsetTopicsAndStresses:
    def __init__(self):
        self.topic_strong_words = [[], [], []]
        self.topic_weak_words = [[], [], []]
        self.stress_strong_words = [[], [], []]
        self.stress_weak_words = [[], [], []]

NOT_APPLICABLE = 0
UNKNOWN = 1
FLUID = 2
INVERTED_TOPIC_CANDIDATE = 3
OUT_OF_SYNC = 4
INVERTED_TOPIC = 5
DISCONNECTED = 6

STRONG_TOPIC = 10
WEAK_TOPIC = 11

class SentInfo:
    _fluid_words = [u"admittedly", u"all in all", u"as a result", u"because", u"conversely", u"equally", u"finally",\
               u"for example", u"in a similar", u"in contrast", u"in summary", u"initially", u"last", u"nevertheless",\
               u"once", u"so far", u"such", u"after", u"along these lines", u"as expected", u"before", u"curiously",\
               u"even though", u"first", u"for instance", u"in a way", u"in other words", u"in the first",\
               u"interestingly", u"lastly", u"next", u"regardless", u"specifically", u"surprisingly", u"afterward",\
               u"although", u"as soon as", u"but", u"despite", u"eventually", u"firstly", u"for this reason",\
               u"in comparison", u"in particular", u"in the same way", u"it follows", u"likewise", u"nonetheless",\
               u"similarly", u"still", u"that is why", u"again", u"as a consequence", u"be that as it may",\
               u"consequently", u"during", u"figure", u"following", u"in a certain sense", u"in conclusion", u"in short",\
               u"indeed", u"it is as if", u"meanwhile", u"now", u"so", u"subsequently", u"the first", u"the last",\
               u"this", u"to elaborate", u"the next", u"this is why", u"to explain", u"the reason", u"thus",\
               u"to illustrate", u"then", u"to conclude", u"to put it another way", u"to put it succinctly",\
               u"unexpectedly", u"while", u"to sum up", u"until", u"while", u"to summarize", u"up to now", u"yet",\
               u"ultimately", u"whereas"]
    def __init__(self, sentence, type, tree):
        self.sent = sentence
        self.type = type
        self.topic_strong_words = []
        self.topic_weak_words = []
        self.stress_strong_words = []
        self.stress_weak_words = []
        self._begins_fluid_words = -1
        self.current_offset = 0
        #0 ~ Sn-1; 1 ~ Sn-2; 2 ~ Sn-3 
        self.offset_wordset = OffsetTopicsAndStresses()
        self._tree = next(tree)
    
    def _deleteNonMainClauses(self, tree):
        idxs = tree.treepositions()
        for s in list(tree.subtrees(lambda t: t.label() == u"SBAR" or t.label() == u"SBARQ")):
            idx = -1
            for x in idxs:
                if s == tree[x]:
                    idx = x
                    break
            if idx != -1:
                del tree[idx]
                idxs = tree.treepositions()
        return tree
    
    def _recursive_search(self, tree, subj_list):
        if tree.height() == 2 or (tree.height() == 3 and len(tree.leaves()) == 1):
            subj_list.extend(tree.leaves())
        else:
            for np_idx in range(len(tree)):
                l = tree[np_idx].label()
                target_tags = [u"NP", u"NN", u"NNS", u"PRP", u"CD"]
                if  l in target_tags:
                    self._recursive_search(tree[np_idx], subj_list)    
    
    def _findSubjects(self, tree):
        res = []
        for s in list(tree.subtrees(lambda t: t.label() == u"S")):
            child_labels_list = []
            for i in range(len(s)):
                child_labels_list.append(s[i].label())
            if u"NP" in child_labels_list and u"VP" in child_labels_list:
                for np_idx in range(len(s)):
                    if s[np_idx].label() == u"NP":
                        self._recursive_search(s[np_idx], res)
        return res
    
    def _appearsBeforeFirstPunct(self, idx):
        punctuation_marks = [tuple([u",", u","]), tuple([u":", u":"]), tuple([u";", u";"]), tuple([r'"', r'``'])]
        if ([x for x in punctuation_marks if x in self.sent[:idx]] == []):
            return True
        else:
            return False
        
    def _appearsAfterLastPunctOrConj(self, idx):
        punctuation_marks = [tuple([u",", u","]), tuple([u":", u":"]), tuple([u";", u";"]), tuple([r'"', r'``'])]
        conj_tags = ["VBD", "VBG", "VBN", "VBP", "VBZ"]
        if ([x for x in punctuation_marks if x in self.sent[idx:]] == []) or\
        ([x for x in self.sent[idx:] if x[1] in conj_tags] == []):
            return True
        else:
            return False
        
    def _isMainClauseContainsTopic(self, main_clause_words):
        topic = list(self.topic_strong_words)
        topic.extend(self.topic_weak_words)
        if ([x for x in topic if not x in main_clause_words] == []):
            return True
        else:
            return False
        
    def _isNumber(self, word):
        numbers = ["one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen,\
        fifteen, sixteen, seventeen, eighteen, nineteen, twenty, thirty, fourty, fifty, sixty, seventy, eighty, ninety,\
        hundred, thousand, million, billion"]
        if (word in numbers) or (re.match(r"^[-+]?[0-9]+$", word) != None):
            return True
        else:
            return False
        
    def _appearsAfterConj(self, idx):
        conj_tags = ["VBD", "VBG", "VBN", "VBP", "VBZ"]
        if ([x for x in self.sent[:idx] if x[1] in conj_tags] != []):
            return True
        else:
            return False
        
    def _isStrongStress(self, stress_word, main_clause_words):
        noun_tags = [u"NN", u"NNP", u"NNS", u"NNPS"]
        if stress_word[1] in noun_tags:
            idx = self.sent.index(stress_word)
            if self._appearsBeforeFirstPunct(idx) or self._appearsAfterLastPunctOrConj(idx):
                return True
            if (stress_word in main_clause_words) and self._isMainClauseContainsTopic(main_clause_words) and\
            self._appearsAfterConj(idx):
                return True
            if self._isNumber(self.sent[idx - 1][0]):
                return True
        #verb derived; stress word in main clause
        elif stress_word[0] in main_clause_words:
            return True
        return False
    
    def _addStressWords(self, stress_words, main_clause_words):
        for stress_word in stress_words:
            if self._isStrongStress(stress_word, main_clause_words):
                self.stress_strong_words.append(stress_word[0])
            else:
                self.stress_weak_words.append(stress_word[0])
    
    def setDefaultWordSet(self):
        self.topic_weak_words = []
        self.topic_strong_words = []
        self.stress_weak_words = []
        self.stress_strong_words = []
        tree = self._tree.copy(True)           
        #display(tree)
        
        #add all words with labels NN, NNS, NNP, NNPS
        #find difference verb derived nouns in VBG FIX
        target_tags = [u"NN", u"NNP", u"NNS", u"NNPS", u"VBG"]
        nounsAndVerbDerivedNouns = []
        for tup in self.sent:
            if tup[1] in target_tags:
                nounsAndVerbDerivedNouns.append(tup)
        #print "nounsAndVerbDerivedNouns: ", nounsAndVerbDerivedNouns, "\n"
        
        #Delete non main clauses
        tree = self._deleteNonMainClauses(tree)
        #Find subjects
        self.topic_strong_words = self._findSubjects(tree)
        stress_words = [x for x in nounsAndVerbDerivedNouns if x[0] not in self.topic_strong_words]
        self._addStressWords(stress_words, tree.leaves())
        #print "topic_strong_words(main clause subj): ", self.topic_strong_words, "\n"
        #print "strong stress: ", self.stress_strong_words, "\n"
        #print "weak stress: ", self.stress_weak_words, "\n"
        #print "current sentence: ", self.sent, "\n"
    
    def addStressWords(self, stress_words, offset):
        tree = self._tree.copy(True)
        tree = self._deleteNonMainClauses(tree)
        #print "stress_words:"
        #print stress_words
        for stress_word in stress_words:
            if self._isStrongStress(stress_word, tree.leaves()):
                self.offset_wordset.stress_strong_words[offset - 1].append(stress_word[0])
            else:
                self.offset_wordset.stress_weak_words[offset - 1].append(stress_word[0])
    
    def getMainClauseSubjects(self):
        tree = self._tree.copy(True)
        tree = self._deleteNonMainClauses(tree)
        #display(tree)
        return self._findSubjects(tree) 
    
    def getMainClauseWords(self):
        tree = self._tree.copy(True)
        tree = self._deleteNonMainClauses(tree)
        return tree.leaves()
        
    def beginsWithFluidWords(self):
        if self._begins_fluid_words != -1:
            return self._begins_fluid_words
        verb_tags = [u"VB", u"VBD", u"VBN", u"VBP", u"VBZ"]
        ordinal_numbers = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",\
                           "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",\
                           "seventeenth", "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth",\
                           "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth"]
        pronouns_tags = [u"PRP$", u"PRP"]
        patt = re.compile(r"([A-Z]+\))|(\([A-Z]+\))|([1-9]+\))|(\([1-9]+\))|([1-9]*1st)|([1-9]*2nd)|([1-9]*3rd)|([1-9]*[4-9]th)|([1-9]+0th)")
        for word in self.sent:
            if word[1] in verb_tags:
                break
            #FIX some FLUID WORDS consist of two and more words, but I consider only one
            if (word[0].lower() in self._fluid_words) or (word[1] in pronouns_tags) or\
            (re.search(patt, word[0]) != None) or (word[0].lower() in ordinal_numbers):
                self.beginsWithFluidWords = 1
            else:
                self.beginsWithFluidWords = 0
        return self.beginsWithFluidWords


def BetweenFluidOrInverted(sent_list, offset):
    #check Sn-1
    if sent_list[-2].type != FLUID and sent_list[-2].type != INVERTED_TOPIC_CANDIDATE:
        return False
    #check Sn-2
    if sent_list[-3].type != FLUID and sent_list[-3].type != INVERTED_TOPIC_CANDIDATE:
        return False
    if offset == 3:
        if sent_list[-4].type != FLUID and sent_list[-4].type != INVERTED_TOPIC_CANDIDATE:
            return False
    return True
    
def TopicFound(topic_words, topic_type, sentence, previous_sentence, reached_verb, offset, sent_list):
#     print "trying to find topic!", topic_words, topic_type
    if offset == 1:
        if not reached_verb:
            sentence.type = FLUID
        else:
            sentence.type = INVERTED_TOPIC_CANDIDATE
    else:
        if not reached_verb:
            if BetweenFluidOrInverted(sent_list, offset):
                sentence.type = FLUID
            else:
                sentence.type = OUT_OF_SYNC
    if topic_type == WEAK_TOPIC:
        if not set(topic_words).issubset(set(sentence.offset_wordset.topic_strong_words[offset - 1])):
            sentence.offset_wordset.topic_weak_words[offset - 1].extend(topic_words)
    elif topic_type == STRONG_TOPIC:
        sentence.offset_wordset.topic_strong_words[offset - 1].extend(topic_words)
                 
def CheckSentenceMainClauses(sentence, previous_sentence, sent_list):
    offset = sentence.current_offset
    main_clause_subjects = sentence.getMainClauseSubjects()
    prev_sent_wordset = list(previous_sentence.topic_weak_words)
    prev_sent_wordset.extend(previous_sentence.topic_strong_words)
    prev_sent_wordset.extend(previous_sentence.stress_strong_words)
#     print "prev_sent_wordset: ", prev_sent_wordset
#     print "main clause subjects: ", main_clause_subjects
    matched_words = [x for x in main_clause_subjects if x in prev_sent_wordset]
#     print "found strong topic! Give me sec to check: ", matched_words
    if matched_words != []:
        TopicFound(matched_words, STRONG_TOPIC, sentence, previous_sentence, False, offset, sent_list)
    stress_words = [x for x in main_clause_subjects if x not in matched_words]
    sentence.addStressWords([(x, u"") for x in stress_words], offset)
    
def CheckWholeSentence(sentence, previous_sentence, sent_list):
    offset = sentence.current_offset
    reached_verb = False
    reached_topic_or_main = False
    prev_sent_wordset = list(previous_sentence.topic_weak_words)
    prev_sent_wordset.extend(previous_sentence.topic_strong_words)
    prev_sent_wordset.extend(previous_sentence.stress_strong_words)
    main_clause_words = sentence.getMainClauseWords()
#     print "prev_sent_wordset: ", prev_sent_wordset
#     print "main clause words: ", main_clause_words
    conj_tags = ["VBD", "VBG", "VBN", "VBP", "VBZ"]
    noun_tags = [u"NN", u"NNP", u"NNS", u"NNPS"]
    
    for word in sentence.sent:
        reached_verb = word[1] in conj_tags or reached_verb
        reached_topic_or_main = word[0] in main_clause_words or reached_topic_or_main
#         print word, reached_verb, reached_topic_or_main
        if not reached_verb:
            matches = word[0] in prev_sent_wordset
            if (matches and word[1] in noun_tags) or (matches and reached_topic_or_main and (word[1] == u"VBG")):
                TopicFound([word[0]], WEAK_TOPIC, sentence, previous_sentence, False, offset, sent_list)
            elif (word[1] == u"VBG"):
                sentence.addStressWords([word], offset)
        else:
            if (sentence.offset_wordset.topic_strong_words[offset - 1] != []) or (sentence.offset_wordset.topic_weak_words[offset - 1] != []):
                if word[1] in noun_tags:
                    sentence.addStressWords([word], offset)
            else:
                matches = word[0] in prev_sent_wordset
                if (word[1] in noun_tags) and matches:
#                     print "special for recognition!!!", [word[0]]
                    TopicFound([word[0]], WEAK_TOPIC, sentence, previous_sentence, True, offset, sent_list)
                    reached_topic_or_main = True
                elif (word[1] == u"VBG") and matches:
                    sentence.addStressWords([word], offset)
                    
def CheckSentenceProgression(sentence, previous_sentence, sent_list):
    sentence.current_offset += 1
    CheckSentenceMainClauses(sentence, previous_sentence, sent_list)
    CheckWholeSentence(sentence, previous_sentence, sent_list)
    
def DefineResults(sentence, total_amount):
    if sentence.type == UNKNOWN:
        sentence.type = DISCONNECTED
        sentence.setDefaultWordSet()
    else:
        if sentence.type == INVERTED_TOPIC:
            word_set_from_round = 1
        elif sentence.type in [FLUID, OUT_OF_SYNC]:
            word_set_from_round = sentence.current_offset
        sentence.topic_strong_words = sentence.offset_wordset.topic_strong_words[word_set_from_round - 1]
        sentence.topic_weak_words = sentence.offset_wordset.topic_weak_words[word_set_from_round - 1]
        sentence.stress_strong_words = sentence.offset_wordset.stress_strong_words[word_set_from_round - 1]
        sentence.stress_weak_words = sentence.offset_wordset.stress_weak_words[word_set_from_round - 1]
    if sentence.type == FLUID:
        total_amount[0] += 1
    elif sentence.type == INVERTED_TOPIC:
        total_amount[1] += 1
    elif sentence.type == OUT_OF_SYNC:
        total_amount[2] +=1
    elif sentence.type == DISCONNECTED:
        total_amount[3] += 1
        
def TypeToString(type):
    if type == FLUID:
        return "Fluid"
    elif type == INVERTED_TOPIC:
        return "Inverted topic"
    elif type == OUT_OF_SYNC:
        return "Out if sync"
    elif type == DISCONNECTED:
        return "Disconnected"
    elif type == NOT_APPLICABLE:
        return "Not applicable"
    
def PrintSentInfo(sent):
    print "--------------------------------------------------------------------------------------------------------"
    print "\nSentence: ", sent.sent
    print "Topic:"
    print "Strong: ", sent.topic_strong_words
    print "Weak  : ", sent.topic_weak_words
    print "Stress:"
    print "Strong: ", sent.stress_strong_words
    print "Weak  : ", sent.stress_weak_words
    print "\nType: ", TypeToString(sent_list[-1].type)
    print "\nWordsets from checkups with Sn-1, Sn-2, Sn-3:"
    print "Topic:"
    print "Strong: ", sent.offset_wordset.topic_strong_words
    print "Weak: ", sent.offset_wordset.topic_weak_words
    print "Stress:"
    print "Strong: ", sent.offset_wordset.stress_strong_words
    print "Weak: ", sent.offset_wordset.stress_weak_words
    print "--------------------------------------------------------------------------------------------------------"
time_beg = datetime.datetime.now()
with codecs.open(sys.argv[1], 'r', 'utf-8') as fin:
        text = fin.read()
paragraphs = PreProcess(text)

total_amount = [0, 0, 0, 0] #fluid, inverted_topic, out_of_sync, disconnected
state = 0
text = SourceText(paragraphs)

plain_paragraphs = []
for par in paragraphs:
    plain_paragraphs.extend(par)
parser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#print "build trees time:"
#time_beg = datetime.datetime.now()
try:
  all_trees = parser.tagged_parse_sents(plain_paragraphs)
except Exception:
  sys.exit()
#print (datetime.datetime.now() - time_beg).total_seconds()

while state == 0:
    #print "\nNext Paragraph:"
    sent_list = []
    sent, p = text.nextSent()
    if sent == []:
        text.nextPar()
        continue
    sent_list.append(SentInfo(sent, NOT_APPLICABLE, next(all_trees)))
    sent_list[-1].setDefaultWordSet()
    #PrintSentInfo(sent_list[-1])
    sent, p = text.nextSent()
    while p == 0:
        sent_list.append(SentInfo(sent, UNKNOWN, next(all_trees)))
        if sent_list[-1].beginsWithFluidWords():
            sent_list[-1].type = FLUID
            #print "setDefaultWordSet:"
            #time_beg = datetime.datetime.now()
            sent_list[-1].setDefaultWordSet()
            #print (datetime.datetime.now() - time_beg).total_seconds()
        else:
            CheckSentenceProgression(sent_list[-1], sent_list[-2], sent_list)
            if sent_list[-1].type == INVERTED_TOPIC_CANDIDATE:
                sent_list[-1].type = INVERTED_TOPIC
            elif sent_list[-1].type == UNKNOWN:
                if len(sent_list) > 2:
                    for offset in range(2):
                        CheckSentenceProgression(sent_list[-1], sent_list[-2], sent_list)
                        if sent_list[-1].type != UNKNOWN or len(sent_list) < 4:
                            break
            DefineResults(sent_list[-1], total_amount)
        #PrintSentInfo(sent_list[-1])
        sent, p = text.nextSent()
    state = text.nextPar()
    #print "############################################################################################################"

total_time = (datetime.datetime.now() - time_beg).total_seconds()
with open(sys.argv[2], "a") as myfile:
    myfile.write(ntpath.basename(sys.argv[1]) + " " + str(total_amount[0]) + " " + str(total_amount[1]) + " " + str(total_amount[2]) + " " + str(total_amount[3]) + "\n")
with open(sys.argv[3], "a") as myfile:
    myfile.write(ntpath.basename(sys.argv[1]) + " " + str(total_time) + "\n")
#print "Total amount:"
#print "Fluid sentences: ", total_amount[0]
#print "Inverted topic sentences: ", total_amount[1]
#print "Out of sync sentences: ", total_amount[2]
#print "Disconnected sentences: ", total_amount[3]
