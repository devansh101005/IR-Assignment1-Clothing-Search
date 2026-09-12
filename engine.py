# Shared layer between the index and the interfaces. The CLI, the GUI and
# the test script all go through this so they cannot disagree about what a
# query means. Also holds the WITHIN/k query parser.

import os
import re

import indexer
import novelty
import positional
import vsm

DEFAULT_CORPUS = os.path.join("data", "corpus_100.txt")


class ClothingSearchEngine(object):

    def __init__(self, corpus_path=DEFAULT_CORPUS, alpha=0.5):
        self.corpus_path = corpus_path
        self.alpha = alpha
        self.index = indexer.build_index(corpus_path)

    def stats(self):
        total_postings = 0
        for term in self.index.inverted:
            total_postings = total_postings + len(self.index.inverted[term])
        return {
            "documents": self.index.N,
            "terms": len(self.index.inverted),
            "postings": total_postings,
        }

    def describe(self, docid):
        return "%s | %s" % (self.index.category_of(docid),
                            self.index.title_of(docid))

    # --- free text ----------------------------------------------------

    def search(self, query_string, top_k=10, use_boost=True, autocorrect=True):
        """use_boost False gives the plain Part B ranking, True adds the
        proximity boost."""

        original = query_string
        changes = []
        dropped = []

        if autocorrect:
            fixed, changes, dropped = novelty.correct_query(self.index,
                                                            query_string)
            query_string = fixed

        if use_boost:
            raw, info = novelty.rank_pbvsm(self.index, query_string, top_k,
                                           self.alpha)
            results = raw
        else:
            plain, info = vsm.rank(self.index, query_string, top_k)
            results = []
            for docid, score in plain:
                results.append({
                    "docid": docid,
                    "base": score,
                    "boost": 1.0,
                    "score": score,
                    "span": 0,
                    "matched": 0,
                    "asked": 0,
                    "factor": 0.0,
                })

        for r in results:
            r["title"] = self.index.title_of(r["docid"])
            r["category"] = self.index.category_of(r["docid"])

        return {
            "original_query": original,
            "used_query": query_string,
            "total_matched": info.get("matched_docs", len(results)),
            "corrections": changes,
            "not_found": dropped,
            "unknown_terms": info["unknown"],
            "zero_idf_terms": info["zero_idf"],
            "fallback": info["fallback"],
            "stems": info["terms"],
            "results": results,
        }

    def explain(self, query_string, docid):
        return vsm.explain(self.index, query_string, docid)

    # --- phrase and proximity ------------------------------------------

    def phrase(self, phrase_string, max_docs=10):
        hits, terms = positional.phrase_search(self.index, phrase_string)

        results = []
        for docid, starts in hits:
            results.append({
                "docid": docid,
                "title": self.index.title_of(docid),
                "category": self.index.category_of(docid),
                "positions": starts,
                "occurrences": len(starts),
                "snippet": positional.phrase_snippet(self.index, docid,
                                                     starts[0], len(terms)),
            })

        # a phrase match is yes/no, but showing the document that repeats
        # the phrase three times before the one that says it once is more
        # useful than plain docID order
        results.sort(key=lambda r: (-r["occurrences"], r["docid"]))

        return {
            "phrase": phrase_string,
            "stems": terms,
            "total": len(hits),
            "results": results[:max_docs],
        }

    def proximity(self, term1, term2, k, ordered=True, max_docs=10):
        hits, pair = positional.proximity_search(self.index, term1, term2, k,
                                                 ordered)

        results = []
        for docid, pairs in hits:
            best = None
            for p1, p2 in pairs:
                gap = abs(p2 - p1)
                if best is None or gap < best:
                    best = gap
            results.append({
                "docid": docid,
                "title": self.index.title_of(docid),
                "category": self.index.category_of(docid),
                "pairs": pairs,
                "closest": best,
                "snippet": positional.snippet_around(self.index, docid,
                                                     pairs[0][0], 6),
            })

        results.sort(key=lambda r: (r["closest"], r["docid"]))

        return {
            "term1": term1,
            "term2": term2,
            "k": k,
            "ordered": ordered,
            "stems": pair,
            "total": len(hits),
            "results": results[:max_docs],
        }

    # --- query parsing --------------------------------------------------

    def parse_positional_query(self, text):
        """cotton WITHIN/3 shirt -> proximity, anything else -> phrase.
        Surrounding quotes are optional."""
        text = text.strip()

        m = re.search(r"^(.*?)\s+WITHIN\s*/\s*(\d+)\s+(.*)$", text,
                      re.IGNORECASE)
        if m is not None:
            left = m.group(1).strip()
            k = int(m.group(2))
            right = m.group(3).strip()
            return "proximity", (left, right, k)

        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'") and len(text) > 1:
            text = text[1:-1]

        return "phrase", (text,)

    def run_positional_query(self, text, max_docs=10):
        kind, args = self.parse_positional_query(text)
        if kind == "proximity":
            left, right, k = args
            out = self.proximity(left, right, k, True, max_docs)
            out["kind"] = "proximity"
            return out
        out = self.phrase(args[0], max_docs)
        out["kind"] = "phrase"
        return out

    def compare(self, query_string, top_k=10):
        return novelty.compare_rankings(self.index, query_string, top_k,
                                        self.alpha)


if __name__ == "__main__":
    eng = ClothingSearchEngine()
    print(eng.stats())
    out = eng.search("cotton shirt")
    for r in out["results"][:5]:
        print("%s %.4f %s" % (r["docid"], r["score"], r["title"]))
    print()
    print(eng.parse_positional_query("cotton WITHIN/3 shirt"))
    print(eng.parse_positional_query('"stretch denim"'))
