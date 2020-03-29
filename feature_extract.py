"""

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

"""

import sys
from util import isStopWord, stemming, punctuation, __PUNCTUATIONS__, CONTRACTION_MAP, CLASS_MAPPINGS
import pickle
from tqdm import tqdm
from math import log10
import re
import os
from collections import Counter
from doc import Document
from constants import __PROB_IDF__, __AUG_TF__, __LOG_AVG__, __LOG_TF__, __IDF__


class Posting:
    def __init__(self, docID, label):
        self.docID = docID
        self.positions = []
        self.tf_idf = None
        self.max_occuring_word_tf = 0
        self.avg_word_tf = 0
        self.class_label = label

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self, type=__LOG_TF__):
        """
        return the term frequency in the document
        :param type: tf variant
        :return: term frequency
        """

        # using log scaled tf to avoid large documents having side effect on the score
        # weighted tf
        term_frequency = len(self.positions)
        tf = 0
        if type.lower() == __LOG_TF__:
            if term_frequency > 0:
                tf = 1 + log10(term_frequency)
            else:
                tf = 0
        elif type.lower() == __AUG_TF__:
            tf = 0.5 + (0.5 * (term_frequency / self.max_occuring_word_tf))
        elif type.lower() == __LOG_AVG__:
            tf = (1 + log10(term_frequency)) / (1 + log10(self.avg_word_tf))
        return tf

    def __str__(self):
        return "(Doc ID: {}, Positions: {}, Term frequency: {})".format(self.docID, self.positions, self.term_freq())

    def __repr(self):
        return self.__str__()


class IndexItem:
    def __init__(self, id, term):
        self.feature_id = id
        self.term = term
        self.posting = {}  # postings are stored in a python dict for easier index building
        self.sorted_postings = []  # may sort them by docID for easier query processing
        self.df = 0

    def add(self, docid, pos, label, max_tf=0, avg_word_tf=0, ):
        """
        add a posting
        :param docid: doc id
        :param pos: term position
        :param label: doc label
        :param max_tf: ignore - for future use
        :param avg_word_tf: ignore - for future use
        :return:
        """
        if not docid in self.posting:
            self.posting[docid] = Posting(docid, label)
            # once per word
            self.df += 1
        self.posting[docid].append(pos)
        self.posting[docid].max_occuring_word_tf = max_tf
        self.posting[docid].avg_word_tf = avg_word_tf

        return self


class InvertedIndex:

    def __init__(self):
        self.feature_id = 0
        self.items = {}  # list of IndexItems
        self.nDocs = 0  # the number of indexed documents

    def indexDoc(self, doc):  # indexing a Document object
        """
        indexing a docuemnt, using the simple SPIMI algorithm,
        but no need to store blocks due to the small collection we are handling.
        Using save/load the whole index instead
        :param doc:
        :return:

        # (1) convert to lower cases,
        # (2) remove stopwords,
        # (3) stemming
        # case conversion
        """

        # convert to lower cases,
        # index size with author 6803 keys and without author
        if not len(doc.body) > 1:
            print("doc {} with body {} ignored from index".format(doc.docID, doc.body))
            return self
        __doc = doc.body.lower()
        # tokenize
        # raw_tokens = __doc.split()
        # https://stackoverflow.com/questions/16926870/split-on-either-a-space-or-a-hyphen
        raw_tokens = re.split("[\s-]+", __doc)
        # strip stopwords, empty spaces and punctuations
        # strip of empty space and punctuations inside the words
        tokens = [self.custom_strip(CONTRACTION_MAP.get(token, token)) for token in raw_tokens]
        # new lines are converted to empty string and strip doenst work here
        tokens = [token for token in tokens if not isStopWord(token) and not punctuation(token) and token != ""]
        # porter stemming
        terms = [stemming(token) for token in tokens if len(token) > 1]
        counts = Counter(terms).values()
        max_tf = max(counts)
        # type conversion for backward compatibility with 2.7
        avg_word_tf = sum(counts) / float(len(counts))
        # postings
        for pos, term in enumerate(terms):
            if term in self.items:
                self.items[term].add(int(doc.docID), pos, label=class_label, max_tf=max_tf, avg_word_tf=avg_word_tf)
            else:
                self.feature_id = self.feature_id + 1
                self.items[term] = IndexItem(self.feature_id, term).add(int(doc.docID), pos, label=class_label,
                                                                        max_tf=max_tf, avg_word_tf=avg_word_tf)
        self.nDocs += 1
        return terms

    @staticmethod
    def custom_strip(self, token):
        return token.strip(__PUNCTUATIONS__ + " ")

    def find(self, term):
        """
        lookup a term in the index
        :param term:
        :return: postings
        """
        return self.items[term]

    def save(self, filename):
        """
        save index file to disk
        :param filename: indexfile name
        :return: None
        """
        # ToDo: using your preferred method to serialize/deserialize the index
        try:
            with open(filename, "wb") as fd:
                pickle.dump(self, fd)
                print("Index saved to ", filename, "successfully!")
        except Exception as e:
            print("Error while saving index. \n Reason: ", e)

    def load(self, filename):
        """
        load index from disk
        :param filename: index file name
        :return: self
        """

        try:
            with open(filename, "rb") as fd:
                invertedIndex = pickle.load(fd)
                self.items = invertedIndex.items
                self.nDocs = invertedIndex.nDocs
                print("Index loaded successfully from ", filename)
        except Exception as e:
            print("Error while loading index. \n Reason: ", e)
        return self

    def idf(self, term, type=__IDF__):
        """
        compute the inverted document frequency for a given term
        :param term: term
        :param type: optional different IDF formula
        :return: idf score
        """
        # ToDo: return the IDF of the term
        try:
            df = 0
            if type.lower() == __IDF__:
                df = self.items[term].df
            elif type.lower() == __PROB_IDF__:
                df = max([0, log10((self.nDocs - self.items[term].df) / self.items[term].df)])

        except KeyError as e:
            # print("Key {} not found".format(term))
            # if no key in index, idf is 0
            return 0
        total_docs = self.nDocs
        return log10(total_docs / float(df))

    def pre_compute_tf_idf(self, type_tf=__LOG_TF__, type_idf=__IDF__):
        """
        pre computes tf-idf scores for each term in the index
        :param type_tf: TF variant
        :param type_idf: IDF variant
        :return: self
        """
        for term in tqdm(self.items, ascii=True, desc="pre-computing tfidf"):
            idf = self.idf(term, type=type_idf)
            for docId in self.items[term].posting:
                posting = self.items[term].posting[docId]
                self.items[term].posting[docId].tfidf = posting.term_freq(type=type_tf) * idf
        return self


def parse_document(filename, _class):
    """
    parse the mini-group document to extract subject and body from each file
    :param filename: name of the document
    :param _class: class_label of a document
    :return:
    """
    with open(filename, "r", encoding="ascii", errors="ignore") as fd:
        # id is the filename
        doc_id = filename.split("/")[-1]
        line_no = -1
        lines = fd.readlines()
        subject = ""
        for line in lines:
            # to avoid any exception for files with no lines
            line_no += 1
            if line.startswith("Subject:"):
                subject = " ".join(line.strip().split(":")[1:])
            if line == "\n":
                break

        body = subject + " ".join([line.strip() for line in lines[line_no:]])
        document = Document(docid=doc_id, body=body, class_label=_class)
    return document


def stringify(features):
    """
    Convert the feature dict to a string to serialize
    :param features:
    :return: str
    """
    content = ""
    for feature_id, score in features:
        content += "{}:{:.5f} ".format(feature_id, score)
    return content.rstrip(" ")


def get_program_args(sys):
    if len(sys.argv) == 1:
        print("Program needs args")
        sys.exit()
    assert len(sys.argv[1:]) == 4
    return sys.argv[1:]


if __name__ == '__main__':
    """
    Usage: python feature_extract.py directory_of_newsgroups_data feature_definition_file class_definition_file training_data_file
    """
    MINI_GROUPS_LOC, feature_def_file, class_def_file, training_data_file = get_program_args(sys)

    indexed_docs = []
    inverted_index = InvertedIndex()
    for news_grp in os.listdir(MINI_GROUPS_LOC):
        news_grp_path = os.path.join(MINI_GROUPS_LOC, news_grp)
        if not news_grp.startswith("."):
            # to avoid files .DS_STORE and other files which are not related
            class_label = CLASS_MAPPINGS[news_grp]
            # parsing and indexing the docs
            for _file in tqdm(os.listdir(news_grp_path), ascii=True, desc="Indexing news group {}".format(news_grp)):
                doc_path = os.path.join(news_grp_path, _file)
                doc = parse_document(doc_path, class_label)
                terms = inverted_index.indexDoc(doc)
                indexed_docs.append((doc, terms))
                # print(doc)
    # create training files
    training_tf_file = "{}.TF".format(training_data_file)
    training_idf_file = "{}.IDF".format(training_data_file)
    training_tf_idf_file = "{}.TFIDF".format(training_data_file)

    # write the class_definition_file
    with open(class_def_file, "w") as fd:
        for grp, label in CLASS_MAPPINGS.items():
            fd.write("{}, {}\n".format(grp, label))
    print(class_def_file, " created successfully! ")

    # write the fetaure_definition_file
    with open(feature_def_file, "w") as fd:
        for item in inverted_index.items:
            fd.write("{}, {}\n".format(inverted_index.items[item].feature_id, item))
    print(feature_def_file, " created successfully! ")

    tf_file = open(training_tf_file, "w")
    idf_file = open(training_idf_file, "w")
    tf_idf_file = open(training_tf_idf_file, "w")

    inverted_index.pre_compute_tf_idf()
    tf_features, idf_features, tf_idf_features = {}, {}, {}
    doc_feature_tf, doc_feature_idf, doc_feature_tf_idf = {}, {}, {}

    for doc, index in tqdm(indexed_docs, ascii=True, desc="extracting features"):
        for term in index:
            fid = inverted_index.items[term].feature_id
            tf_features[fid] = inverted_index.items[term].posting[doc.docID].term_freq(type=__LOG_TF__)
            idf_features[fid] = inverted_index.idf(term)
            tf_idf_features[fid] = tf_features[fid] * idf_features[fid]
        tf_features = sorted(tf_features.items(), key=lambda x: x[0])
        idf_features = sorted(idf_features.items(), key=lambda x: x[0])
        tf_idf_features = sorted(tf_idf_features.items(), key=lambda x: x[0])
        tf_file.write("{} {}\n".format(doc.class_label, stringify(tf_features)))
        idf_file.write("{} {}\n".format(doc.class_label, stringify(idf_features)))
        tf_idf_file.write("{} {}\n".format(doc.class_label, stringify(tf_idf_features)))
        tf_features, idf_features, tf_idf_features = {}, {}, {}

    tf_file.close()
    print(training_tf_file, " successfully created!")
    idf_file.close()
    print(training_idf_file, " successfully created!")
    tf_idf_file.close()
    print(training_tf_idf_file, " successfully created!")
