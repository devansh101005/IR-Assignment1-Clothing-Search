# Reads the corpus and builds both indexes:
#   inverted   term -> df -> [(docID, tf), ...]            (Part A / B)
#   positional term -> df -> [(docID, tf, [p1, p2...]), ...] (Part C)
# tf is really just len(positions), but keeping the two structures apart
# makes the two output dumps readable and lets us compare retrieval with
# and without positions.

import math
import re

import preprocess


class SearchIndex(object):

    def __init__(self):
        self.docs = {}          # docID -> category / title / text
        self.doc_order = []     # docIDs in file order
        self.inverted = {}      # term -> {docID: tf}
        self.positional = {}    # term -> {docID: [positions]}
        self.df = {}
        self.doc_length = {}    # docID -> cosine normalisation length
        self.doc_tokens = {}    # docID -> terms kept after preprocessing
        self.N = 0

    def load_corpus(self, path):
        """Pull the fields out of the SGML style corpus file. A real XML
        parser would choke on the unescaped % and apostrophes."""

        f = open(path, "r", encoding="utf-8")
        raw = f.read()
        f.close()

        blocks = re.findall(r"<DOC>(.*?)</DOC>", raw, re.S)
        if len(blocks) == 0:
            raise ValueError("no <DOC> blocks found in " + path)

        for block in blocks:
            docid = self._field(block, "DOCID")
            self.docs[docid] = {
                "category": self._field(block, "CATEGORY"),
                "title": self._field(block, "TITLE"),
                "text": self._field(block, "TEXT"),
            }
            self.doc_order.append(docid)

        self.N = len(self.doc_order)
        return self.N

    def _field(self, block, tag):
        m = re.search("<" + tag + ">(.*?)</" + tag + ">", block, re.S)
        if m is None:
            return ""
        return m.group(1).strip()

    def build(self):
        """One pass over the collection fills both indexes and the dfs."""

        for docid in self.doc_order:
            d = self.docs[docid]

            # category and title are indexed with the body. They repeat the
            # important words, so those terms get a higher tf. Indexing the
            # body alone gave clearly worse rankings for "black t-shirt".
            full_text = d["category"] + " " + d["title"] + " " + d["text"]

            terms = preprocess.analyse(full_text)
            self.doc_tokens[docid] = len(terms)

            for position, term in terms:
                if term not in self.positional:
                    self.positional[term] = {}
                if docid not in self.positional[term]:
                    self.positional[term][docid] = []
                self.positional[term][docid].append(position)

                if term not in self.inverted:
                    self.inverted[term] = {}
                if docid not in self.inverted[term]:
                    self.inverted[term][docid] = 0
                self.inverted[term][docid] = self.inverted[term][docid] + 1

        for term in self.inverted:
            self.df[term] = len(self.inverted[term])

        self._compute_doc_lengths()

    def _compute_doc_lengths(self):
        """Document vector lengths for the cosine normalisation.
        For lnc the weight is 1 + log10(tf) with no idf, so

            |d| = sqrt( sum (1 + log10(tf))^2 )

        Computed once here and reused for every query."""

        totals = {}
        for docid in self.doc_order:
            totals[docid] = 0.0

        for term in self.inverted:
            for docid in self.inverted[term]:
                w = 1.0 + math.log10(self.inverted[term][docid])
                totals[docid] = totals[docid] + (w * w)

        for docid in self.doc_order:
            self.doc_length[docid] = math.sqrt(totals[docid])

    def idf(self, term):
        """log10(N/df). Unknown term gives 0.0 so a missing word just
        contributes nothing instead of raising."""
        if term not in self.df:
            return 0.0
        return math.log10(float(self.N) / float(self.df[term]))

    def term_frequency(self, term, docid):
        if term not in self.inverted:
            return 0
        if docid not in self.inverted[term]:
            return 0
        return self.inverted[term][docid]

    def postings(self, term):
        """(docID, tf) pairs, sorted by docID."""
        if term not in self.inverted:
            return []
        out = []
        for docid in sorted(self.inverted[term]):
            out.append((docid, self.inverted[term][docid]))
        return out

    def positional_postings(self, term):
        """(docID, tf, [positions]) triples, sorted by docID."""
        if term not in self.positional:
            return []
        out = []
        for docid in sorted(self.positional[term]):
            plist = self.positional[term][docid]
            out.append((docid, len(plist), plist))
        return out

    def vocabulary(self):
        return sorted(self.inverted.keys())

    def title_of(self, docid):
        return self.docs[docid]["title"]

    def category_of(self, docid):
        return self.docs[docid]["category"]

    # --- output dumps ------------------------------------------------

    def dump_inverted_index(self, path):
        f = open(path, "w", encoding="utf-8")
        f.write("INVERTED INDEX (Part A / Part B)\n")
        f.write("Collection size N = %d documents\n" % self.N)
        f.write("Dictionary size   = %d distinct terms\n" % len(self.inverted))
        f.write("\nFormat:  term  ->  df  ->  [(docID, tf), ...]\n")
        f.write("=" * 100 + "\n\n")

        for term in self.vocabulary():
            f.write("%-14s df=%-4d " % (term, self.df[term]))
            f.write("idf=%.4f  " % self.idf(term))
            pairs = []
            for docid, tf in self.postings(term):
                pairs.append("(%s,%d)" % (docid, tf))
            f.write("postings: " + ", ".join(pairs) + "\n")
        f.close()

    def dump_positional_index(self, path):
        f = open(path, "w", encoding="utf-8")
        f.write("POSITIONAL INDEX (Part C)\n")
        f.write("Collection size N = %d documents\n" % self.N)
        f.write("Dictionary size   = %d distinct terms\n" % len(self.positional))
        f.write("\nFormat:  term  ->  df  ->  [(docID, tf, [p1, p2, ...]), ...]\n")
        f.write("Positions are offsets in the token stream of "
                "CATEGORY + TITLE + TEXT, counted before stop word removal.\n")
        f.write("=" * 100 + "\n\n")

        for term in self.vocabulary():
            f.write("%s  ->  df=%d\n" % (term, self.df[term]))
            for docid, tf, plist in self.positional_postings(term):
                positions = []
                for p in plist:
                    positions.append(str(p))
                f.write("        (%s, tf=%d, [%s])\n"
                        % (docid, tf, ", ".join(positions)))
            f.write("\n")
        f.close()

    def dump_dictionary(self, path):
        """term / df / idf / cf sorted by df. Sorting by df is what makes
        the boilerplate problem obvious."""
        f = open(path, "w", encoding="utf-8")
        f.write("DICTIONARY - term, document frequency, idf, collection frequency\n")
        f.write("Sorted by df (highest first). N = %d\n" % self.N)
        f.write("=" * 78 + "\n")
        f.write("%-16s %6s %10s %8s\n" % ("TERM", "df", "idf", "cf"))
        f.write("-" * 78 + "\n")

        rows = []
        for term in self.vocabulary():
            cf = 0
            for docid in self.inverted[term]:
                cf = cf + self.inverted[term][docid]
            rows.append((self.df[term], term, cf))
        rows.sort(key=lambda r: (-r[0], r[1]))

        for df, term, cf in rows:
            f.write("%-16s %6d %10.4f %8d\n" % (term, df, self.idf(term), cf))
        f.close()

    def dump_doc_lengths(self, path):
        f = open(path, "w", encoding="utf-8")
        f.write("DOCUMENT LENGTHS used for cosine normalisation (lnc)\n")
        f.write("length = sqrt( sum of (1 + log10(tf))^2 over all terms in the doc )\n")
        f.write("=" * 90 + "\n")
        f.write("%-8s %8s %10s   %s\n" % ("docID", "#terms", "length", "title"))
        f.write("-" * 90 + "\n")
        for docid in self.doc_order:
            f.write("%-8s %8d %10.5f   %s\n"
                    % (docid, self.doc_tokens[docid], self.doc_length[docid],
                       self.title_of(docid)))
        f.close()


def build_index(corpus_path):
    idx = SearchIndex()
    idx.load_corpus(corpus_path)
    idx.build()
    return idx


if __name__ == "__main__":
    index = build_index("data/corpus_100.txt")
    print("documents indexed :", index.N)
    print("distinct terms    :", len(index.inverted))
    print()
    print("sample postings for 'cotton':")
    print("  ", index.postings("cotton")[:8])
    print("sample positional postings for 'cotton':")
    print("  ", index.positional_postings("cotton")[:3])
