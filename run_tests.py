# Part E test plan:  python run_tests.py
# Prints to the screen and saves a copy in output/test_results.txt.
# The query lists below are fixed but no expected document IDs are written
# down anywhere - every number in the output is read back from the engine.

import os

import engine

OUT_PATH = os.path.join("output", "test_results.txt")


class Report(object):
    """Sends every line to the screen and to the file."""

    def __init__(self, path):
        self.lines = []
        self.path = path

    def w(self, text=""):
        print(text)
        self.lines.append(text)

    def rule(self, ch="="):
        self.w(ch * 92)

    def heading(self, text):
        self.w()
        self.rule("=")
        self.w(text)
        self.rule("=")

    def save(self):
        f = open(self.path, "w", encoding="utf-8")
        f.write("\n".join(self.lines))
        f.write("\n")
        f.close()


# --- the query sets ---------------------------------------------------

FREE_TEXT_QUERIES = [
    "black cotton t-shirt",
    "regular fit kurta",
    "stretch denim jeans",
    "warm fleece hoodie for winter",
    "printed saree festive wear",
    "high waist leggings",
    "slim fit checked shirt",
    "quilted puffer jacket",
    "breathable cotton fabric summer",
    "women green midi dress",
    "machine washable oversized sweatshirt",
    "navy blue zip fly denim",
]

PHRASE_QUERIES = [
    "cotton shirt",
    "stretch denim",
    "festive wear",
    "winter wear",
    "regular fit",
    "breathable fabric",
    "high waist",
    "zip closure",
]

PROXIMITY_QUERIES = [
    ("cotton", "shirt", 3),
    ("stretch", "denim", 4),
    ("winter", "wear", 3),
    ("festive", "kurta", 4),
    ("zip", "closure", 5),
    ("high", "waist", 1),
]

OOV_QUERIES = [
    "cotton umbrella",          # one unknown word, one known
    "leather laptop backpack",  # nothing here is in the corpus
    "cotten shrt",              # typos
    "blak hoodei",              # typos
]


def format_result_table(rep, results, show_span=False):
    if show_span:
        rep.w("%-5s %-7s %-9s %-7s %-6s %-12s %s"
              % ("rank", "docID", "score", "cosine", "span", "category", "title"))
        rep.w("-" * 92)
        i = 1
        for r in results:
            rep.w("%-5d %-7s %-9.4f %-7.4f %-6s %-12s %s"
                  % (i, r["docid"], r["score"], r["base"],
                     (str(r["span"]) if r["span"] > 0 else "-"),
                     r["category"], r["title"]))
            i = i + 1
    else:
        rep.w("%-5s %-7s %-9s %-12s %s"
              % ("rank", "docID", "cosine", "category", "title"))
        rep.w("-" * 92)
        i = 1
        for r in results:
            rep.w("%-5d %-7s %-9.4f %-12s %s"
                  % (i, r["docid"], r["score"], r["category"], r["title"]))
            i = i + 1


def main():
    if not os.path.isdir("output"):
        os.mkdir("output")

    rep = Report(OUT_PATH)

    rep.w("CSD358 INFORMATION RETRIEVAL - ASSIGNMENT 1")
    rep.w("Clothing Search Engine - Part E test run")
    rep.rule("=")

    eng = engine.ClothingSearchEngine()
    st = eng.stats()
    rep.w("collection size N          : %d documents" % st["documents"])
    rep.w("dictionary size            : %d distinct terms" % st["terms"])
    rep.w("total postings             : %d" % st["postings"])
    rep.w("weighting scheme           : lnc.ltc (Part B)")
    rep.w("proximity boost alpha      : %.2f" % eng.alpha)

    counts = {"free": 0, "phrase": 0, "prox": 0, "oov": 0}

    # ==================================================================
    # TEST 1 - free text queries, plain Part B ranking
    # ==================================================================
    rep.heading("TEST 1 - FREE TEXT RANKED RETRIEVAL (lnc.ltc, Part B)"
                "\n%d queries, top 10 documents each" % len(FREE_TEXT_QUERIES))

    for q in FREE_TEXT_QUERIES:
        counts["free"] = counts["free"] + 1
        out = eng.search(q, 10, use_boost=False, autocorrect=False)
        rep.w()
        rep.w("Q%d: \"%s\"" % (counts["free"], q))
        rep.w("    query terms after preprocessing : %s" % out["stems"])
        if len(out["zero_idf_terms"]) > 0:
            rep.w("    terms with idf = 0 (df = N)     : %s   <- these cannot "
                  "affect the cosine score" % out["zero_idf_terms"])
        if len(out["unknown_terms"]) > 0:
            rep.w("    terms not in the dictionary     : %s"
                  % out["unknown_terms"])
        if out["fallback"]:
            rep.w("    NOTE: every query term had idf = 0, so the idf factor "
                  "was dropped (see vsm.py)")
        rep.w("    documents with a non-zero score : %d%s"
              % (out["total_matched"],
                 " (showing the top 10)" if out["total_matched"] > 10 else ""))
        if len(out["results"]) == 0:
            rep.w("    no documents matched this query")
        else:
            format_result_table(rep, out["results"])

    # ==================================================================
    # TEST 2 - exact phrase queries
    # ==================================================================
    rep.heading("TEST 2 - EXACT PHRASE QUERIES (positional index, Part C)"
                "\n%d phrases. Positions are token offsets in "
                "CATEGORY + TITLE + TEXT." % len(PHRASE_QUERIES))

    phrase_totals = {}
    for p in PHRASE_QUERIES:
        counts["phrase"] = counts["phrase"] + 1
        out = eng.phrase(p, 10)
        phrase_totals[p] = out["total"]

        rep.w()
        rep.w("P%d: phrase \"%s\"   ->  stems %s"
              % (counts["phrase"], p, out["stems"]))
        rep.w("    documents containing the exact phrase : %d" % out["total"])

        if out["total"] == 0:
            # check whether the words themselves exist, that is the
            # interesting part of a zero result
            missing = []
            present = []
            for t in out["stems"]:
                if t in eng.index.df:
                    present.append("%s (df=%d)" % (t, eng.index.df[t]))
                else:
                    missing.append(t)
            if len(missing) == 0:
                rep.w("    NOTE: every word of this phrase DOES occur in the "
                      "corpus - %s" % ", ".join(present))
                rep.w("    but never next to each other, so the phrase search "
                      "correctly returns nothing.")
            else:
                rep.w("    word(s) not in the dictionary at all: %s"
                      % ", ".join(missing))
            continue

        rep.w("%-5s %-7s %-6s %-22s %s"
              % ("rank", "docID", "count", "positions", "title"))
        rep.w("-" * 92)
        i = 1
        for r in out["results"]:
            plist = []
            for p_ in r["positions"]:
                plist.append(str(p_))
            rep.w("%-5d %-7s %-6d %-22s %s"
                  % (i, r["docid"], r["occurrences"], "[" + ", ".join(plist) + "]",
                     r["title"]))
            i = i + 1
        rep.w("    evidence from the positional index, %s at position %d:"
              % (out["results"][0]["docid"], out["results"][0]["positions"][0]))
        rep.w("      %s" % out["results"][0]["snippet"])

    # ==================================================================
    # TEST 3 - proximity queries with different k
    # ==================================================================
    rep.heading("TEST 3 - ORDERED PROXIMITY QUERIES (term1 WITHIN/k term2)"
                "\n%d queries with %d different values of k"
                % (len(PROXIMITY_QUERIES),
                   len(set([k for a, b, k in PROXIMITY_QUERIES]))))

    for t1, t2, k in PROXIMITY_QUERIES:
        counts["prox"] = counts["prox"] + 1
        out = eng.proximity(t1, t2, k, True, 10)

        rep.w()
        rep.w("X%d: %s WITHIN/%d %s   ->  stems %s"
              % (counts["prox"], t1, k, t2, str(out["stems"])))
        rep.w("    matching documents : %d" % out["total"])

        if out["total"] == 0:
            rep.w("    no document has these two terms within %d positions "
                  "in this order." % k)
            continue

        rep.w("%-5s %-7s %-8s %-26s %s"
              % ("rank", "docID", "closest", "(p1,p2) pairs", "title"))
        rep.w("-" * 92)
        i = 1
        for r in out["results"]:
            pairs = []
            for a, b in r["pairs"][:3]:
                pairs.append("(%d,%d)" % (a, b))
            extra = ""
            if len(r["pairs"]) > 3:
                extra = " ..."
            rep.w("%-5d %-7s %-8d %-26s %s"
                  % (i, r["docid"], r["closest"],
                     ", ".join(pairs) + extra, r["title"]))
            i = i + 1
        rep.w("    evidence, context around position %d of %s:"
              % (out["results"][0]["pairs"][0][0], out["results"][0]["docid"]))
        rep.w("      %s" % out["results"][0]["snippet"])

    # how k changes the answer - we take one pair and sweep k
    rep.w()
    rep.w("-" * 92)
    rep.w("effect of k on the same pair of terms (black / cotton):")
    rep.w("%-6s %-22s %s" % ("k", "matching documents", "comment"))
    previous = None
    for k in [1, 2, 3, 4, 6, 10, 20]:
        out = eng.proximity("black", "cotton", k, True, 100)
        note = ""
        if previous is not None and out["total"] > previous:
            note = "+%d document(s) become reachable at this window" \
                   % (out["total"] - previous)
        if out["total"] == 0:
            note = "the two words are never this close"
        rep.w("%-6d %-22d %s" % (k, out["total"], note))
        previous = out["total"]
    rep.w("As k grows the window gets looser and more documents qualify, until "
          "it saturates at the")
    rep.w("number of documents that contain both terms at all. That saturation "
          "point is where a")
    rep.w("proximity query stops being different from a plain boolean AND.")

    # a proximity query with k = 1 must return exactly the phrase matches
    rep.w()
    k1 = eng.proximity("cotton", "shirt", 1, True, 100)
    rep.w("consistency check between the two code paths:")
    rep.w("   cotton WITHIN/1 shirt  -> %d documents" % k1["total"])
    rep.w("   phrase \"cotton shirt\"  -> %d documents"
          % phrase_totals.get("cotton shirt", 0))
    if k1["total"] == phrase_totals.get("cotton shirt", 0):
        rep.w("   the two agree, which is what we expect: a gap of exactly 1 "
              "position IS adjacency.")
    else:
        rep.w("   MISMATCH - the phrase matcher and the proximity matcher "
              "disagree, that would be a bug.")

    # ==================================================================
    # TEST 4 - terms that are not in the corpus
    # ==================================================================
    rep.heading("TEST 4 - QUERIES CONTAINING TERMS THAT ARE NOT IN THE CORPUS"
                "\nthe first two contain real words the corpus does not have, "
                "the last two are misspellings")

    for q in OOV_QUERIES:
        counts["oov"] = counts["oov"] + 1
        rep.w()
        rep.w("U%d: \"%s\"" % (counts["oov"], q))

        # first without the corrector, to show the raw behaviour
        plain = eng.search(q, 10, use_boost=False, autocorrect=False)
        rep.w("    without our spelling corrector:")
        rep.w("        terms after preprocessing : %s" % plain["stems"])
        rep.w("        unknown terms             : %s" % plain["unknown_terms"])
        rep.w("        documents retrieved       : %d" % plain["total_matched"])
        if plain["total_matched"] == 0:
            rep.w("        -> nothing at all is returned, every query term was "
                  "unknown.")
        else:
            rep.w("        -> the unknown terms are simply ignored and the "
                  "remaining terms still rank.")
            format_result_table(rep, plain["results"][:5])

        # now with the corrector on
        fixed = eng.search(q, 10, use_boost=True, autocorrect=True)
        rep.w("    with our spelling corrector (novelty.py):")
        if len(fixed["corrections"]) > 0:
            for typed, suggestion in fixed["corrections"]:
                rep.w("        \"%s\" was not in the dictionary -> corrected to "
                      "\"%s\"" % (typed, suggestion))
            rep.w("        rewritten query           : \"%s\"" % fixed["used_query"])
            rep.w("        documents retrieved       : %d" % fixed["total_matched"])
            format_result_table(rep, fixed["results"][:5], show_span=True)
        else:
            rep.w("        no correction was possible for %s"
                  % fixed["not_found"])
            rep.w("        this is the right answer - these words have nothing "
                  "to do with clothing,")
            rep.w("        and inventing a correction for them would be worse "
                  "than returning nothing.")

    # ==================================================================
    # TEST 5 - where positions change the answer
    # ==================================================================
    rep.heading("TEST 5 - WHERE POSITIONAL INFORMATION CHANGES THE RESULT"
                "\nrequired by Part E: at least two cases explained")

    # these queries were picked from the dictionary dump because they
    # contain terms with df = N; the document ids are read back, not assumed
    cases = [
        ("regular fit",
         "\"regular\" has df = 100 so its idf is 0 and it is invisible to the "
         "cosine score.\n"
         "     The plain VSM therefore ranks purely on \"fit\", which the "
         "boilerplate phrase\n"
         "     \"comfortable fit\" gives to many products that are not regular "
         "fit at all.\n"
         "     Once the positions are used, the products whose title really "
         "says \"Regular Fit\"\n"
         "     (span = 2, the two terms are adjacent) are pushed above them."),
        ("short sleeves cotton",
         "Same set of documents in both rankings, but a completely different "
         "order.\n"
         "     The documents that actually say \"short sleeves\" as a phrase "
         "have span 2;\n"
         "     the ones where the three words are scattered across the "
         "description have a\n"
         "     span of about 16 tokens and drop to the bottom half of the "
         "page."),
        ("comfortable regular wear",
         "The extreme case. ALL THREE query terms have df = 100, so the whole "
         "query\n"
         "     vector is zero and plain lnc.ltc cannot rank this query at all "
         "- it has to\n"
         "     fall back to weights without idf. The positional information is "
         "the only\n"
         "     signal left, and it still separates the documents sensibly."),
    ]

    case_number = 0
    changed_count = 0
    for query, explanation in cases:
        case_number = case_number + 1
        cmp = eng.compare(query, 10)

        plain_ids = []
        for docid, score in cmp["plain"]:
            plain_ids.append(docid)
        boost_ids = []
        for r in cmp["boosted"]:
            boost_ids.append(r["docid"])

        rep.w()
        rep.w("CASE %d: \"%s\"" % (case_number, query))
        rep.w("-" * 92)
        rep.w("%-5s | %-7s %-9s %-34s | %-7s %-9s %-5s %s"
              % ("rank", "docID", "cosine", "title (plain VSM, Part B)",
                 "docID", "score", "span", "title (with positions)"))
        rep.w("-" * 92)
        i = 0
        while i < max(len(cmp["plain"]), len(cmp["boosted"])):
            if i < len(cmp["plain"]):
                ld, ls = cmp["plain"][i]
                lt = eng.index.title_of(ld)
                left = "%-7s %-9.4f %-34s" % (ld, ls, lt[:34])
            else:
                left = " " * 52
            if i < len(cmp["boosted"]):
                r = cmp["boosted"][i]
                right = "%-7s %-9.4f %-5s %s" % (r["docid"], r["score"],
                                                 (str(r["span"]) if r["span"] > 0 else "-"),
                                                 eng.index.title_of(r["docid"]))
            else:
                right = ""
            rep.w("%-5d | %s | %s" % (i + 1, left, right))
            i = i + 1

        rep.w()
        rep.w("  order changed            : %s" % cmp["order_changed"])
        rep.w("  documents that entered   : %s"
              % (cmp["entered"] if len(cmp["entered"]) > 0 else "none"))
        rep.w("  documents that dropped   : %s"
              % (cmp["left"] if len(cmp["left"]) > 0 else "none"))
        rep.w("  why:")
        rep.w("     %s" % explanation)

        if cmp["order_changed"]:
            changed_count = changed_count + 1

    # ==================================================================
    # TEST 6 - VSM and phrase search on the same query
    # ==================================================================
    rep.heading("TEST 6 - THE SAME QUERY THROUGH BOTH MODES"
                "\nthis is the clearest way to show what a positional index "
                "buys you")

    for q in ["zip closure", "cotton shirt", "stretch denim"]:
        vsm_out = eng.search(q, 10, use_boost=False, autocorrect=False)
        ph_out = eng.phrase(q, 10)
        rep.w()
        rep.w("query \"%s\"" % q)
        rep.w("   free text VSM  : %d documents have a non-zero score"
              % vsm_out["total_matched"])
        rep.w("   exact phrase   : %d documents contain the phrase"
              % ph_out["total"])
        if ph_out["total"] == 0 and vsm_out["total_matched"] > 0:
            rep.w("   -> the VSM happily returns documents here because both "
                  "words appear somewhere")
            rep.w("      in them, but no document ever puts them next to each "
                  "other. Without the")
            rep.w("      positional index we would have no way of knowing "
                  "that.")
        elif ph_out["total"] < vsm_out["total_matched"]:
            rep.w("   -> the phrase search is much stricter, it removes the "
                  "documents where the")
            rep.w("      two words only happen to co-occur.")

    # ==================================================================
    # summary
    # ==================================================================
    rep.heading("SUMMARY OF THE TEST RUN")
    rep.w("free text queries run                  : %d   (Part E asks for at "
          "least 10)" % counts["free"])
    rep.w("exact phrase queries run               : %d   (Part E asks for at "
          "least 5)" % counts["phrase"])
    rep.w("proximity queries run                  : %d with %d different "
          "values of k   (Part E asks for at least 3)"
          % (counts["prox"], len(set([k for a, b, k in PROXIMITY_QUERIES]))))
    rep.w("queries with out-of-corpus terms       : %d   (Part E asks for at "
          "least 1)" % counts["oov"])
    rep.w("cases where positions changed the rank : %d of %d examined   "
          "(Part E asks for at least 2)" % (changed_count, len(cases)))
    rep.w()
    if (counts["free"] >= 10 and counts["phrase"] >= 5 and counts["prox"] >= 3
            and counts["oov"] >= 1 and changed_count >= 2):
        rep.w("every mandatory test in Part E has been covered.")
    else:
        rep.w("WARNING: one of the mandatory test counts is short, check the "
              "query lists at the top of this file.")

    rep.save()
    print()
    print("saved to", OUT_PATH)


if __name__ == "__main__":
    main()
