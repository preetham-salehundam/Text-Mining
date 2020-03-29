'''

The document class, containing information from the raw document and possibly other tasks

The collection class holds a set of docuemnts, indexed by docID

'''

class Document:
    def __init__(self, docid, class_label, body):
        self.docID = int(docid)
        self.class_label = int(class_label)
        self.body = body

    # add more methods if needed


class Collection:
    ''' a collection of documents'''

    def __init__(self):
        self.docs = {} # documents are indexed by docID

    def find(self, docID):
        ''' return a document object'''
        if self.docs.has_key(docID):
            return self.docs[docID]
        else:
            return None

    # more methods if needed

