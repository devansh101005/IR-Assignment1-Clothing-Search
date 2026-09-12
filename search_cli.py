# Part D, terminal interface:  python search_cli.py
# Mode 1 free text (Part B), mode 2 phrase / proximity (Part C),
# mode 3 both rankings side by side. Type "help" for the commands.

import sys

import engine

BANNER = r"""
+--------------------------------------------------------------------------+
|   CLOTHING SEARCH ENGINE   -   CSD358 Information Retrieval, Assignment 1 |
+--------------------------------------------------------------------------+
"""

HELP = """
SEARCH MODES
   1  or  free        free text search, ranked with lnc.ltc cosine similarity
   2  or  phrase      phrase and proximity search using the positional index
   3  or  compare     show the ranking with and without positional information

IN FREE TEXT MODE just type the query, for example
   black cotton t-shirt
   warm fleece hoodie for winter

IN PHRASE MODE you can type
   cotton shirt              -> exact phrase search
   "stretch denim"           -> the quotes are optional
   cotton WITHIN/3 shirt     -> ordered proximity, the two terms within 3 tokens

OTHER COMMANDS
   stats              size of the collection and of the dictionary
   term <word>        show the postings and the positions stored for one term
   explain <docID>    show the lnc.ltc arithmetic of the last query for one doc
   boost on | off     turn the proximity boost on or off (default on)
   alpha <number>     change how strong the proximity boost is (default 0.5)
   help               this text
   quit               leave
"""


def print_free_text(out, boost_on):
    print()
    if len(out["corrections"]) > 0:
        for typed, fixed in out["corrections"]:
            print('   did you mean "%s" instead of "%s" ?' % (fixed, typed))
        print('   searching for: "%s"' % out["used_query"])
    if len(out["not_found"]) > 0:
        print("   these words are not in the collection and were ignored: %s"
              % ", ".join(out["not_found"]))

    print("   query terms after preprocessing : %s" % out["stems"])
    if len(out["zero_idf_terms"]) > 0:
        print("   terms with idf = 0 (they occur in all %d documents, so they "
              "cannot affect the score): %s"
              % (100, ", ".join(out["zero_idf_terms"])))
    if out["fallback"]:
        print("   NOTE: every term of this query has idf = 0. The plain vector "
              "space model")
        print("         cannot rank this query, try the phrase mode instead.")

    if len(out["results"]) == 0:
        print()
        print("   no matching products.")
        return

    print("   %d documents matched, showing the top %d"
          % (out["total_matched"], len(out["results"])))
    print()

    if boost_on:
        print("   %-4s %-7s %-9s %-9s %-6s %-12s %s"
              % ("#", "docID", "score", "cosine", "span", "category", "title"))
        print("   " + "-" * 88)
        rank = 1
        for r in out["results"]:
            if r["span"] > 0:
                span = str(r["span"])
            else:
                span = "-"
            print("   %-4d %-7s %-9.4f %-9.4f %-6s %-12s %s"
                  % (rank, r["docid"], r["score"], r["base"], span,
                     r["category"], r["title"]))
            rank = rank + 1
        print()
        print("   score  = cosine x (1 + alpha x coverage x tightness)")
        print("   span   = shortest window of the document that contains the "
              "query terms")
    else:
        print("   %-4s %-7s %-9s %-12s %s"
              % ("#", "docID", "cosine", "category", "title"))
        print("   " + "-" * 88)
        rank = 1
        for r in out["results"]:
            print("   %-4d %-7s %-9.4f %-12s %s"
                  % (rank, r["docid"], r["score"], r["category"], r["title"]))
            rank = rank + 1


def print_positional(out):
    print()
    if out["kind"] == "phrase":
        print('   exact phrase "%s"   ->  stems %s' % (out["phrase"], out["stems"]))
        print("   %d document(s) contain this phrase" % out["total"])
        if out["total"] == 0:
            print()
            print("   nothing found. Note that this is NOT the same as the "
                  "words being absent -")
            print("   they may well both be in the collection but never next "
                  "to each other.")
            return
        print()
        print("   %-4s %-7s %-6s %-20s %s"
              % ("#", "docID", "count", "positions", "title"))
        print("   " + "-" * 88)
        rank = 1
        for r in out["results"]:
            plist = []
            for p in r["positions"]:
                plist.append(str(p))
            print("   %-4d %-7s %-6d %-20s %s"
                  % (rank, r["docid"], r["occurrences"],
                     "[" + ", ".join(plist) + "]", r["title"]))
            rank = rank + 1
        print()
        print("   evidence that the positional index is really being used:")
        for r in out["results"][:3]:
            print("     %s at position %d ->  %s"
                  % (r["docid"], r["positions"][0], r["snippet"]))

    else:
        print("   %s WITHIN/%d %s   ->  stems %s"
              % (out["term1"], out["k"], out["term2"], str(out["stems"])))
        print("   %d document(s) have the two terms within %d positions, in "
              "this order" % (out["total"], out["k"]))
        if out["total"] == 0:
            print()
            print("   nothing found at this value of k. Try a larger k.")
            return
        print()
        print("   %-4s %-7s %-8s %-26s %s"
              % ("#", "docID", "closest", "(p1,p2)", "title"))
        print("   " + "-" * 88)
        rank = 1
        for r in out["results"]:
            pairs = []
            for a, b in r["pairs"][:3]:
                pairs.append("(%d,%d)" % (a, b))
            tail = ""
            if len(r["pairs"]) > 3:
                tail = " ..."
            print("   %-4d %-7s %-8d %-26s %s"
                  % (rank, r["docid"], r["closest"], ", ".join(pairs) + tail,
                     r["title"]))
            rank = rank + 1
        print()
        print("   evidence that the positional index is really being used:")
        for r in out["results"][:3]:
            print("     %s first hit at %d ->  %s"
                  % (r["docid"], r["pairs"][0][0], r["snippet"]))


def print_compare(eng, query):
    cmp = eng.compare(query, 10)
    print()
    print("   left  = plain lnc.ltc (Part B)      right = with positions")
    print()
    print("   %-4s %-7s %-9s %-30s  %-7s %-9s %-5s %s"
          % ("#", "docID", "cosine", "title", "docID", "score", "span", "title"))
    print("   " + "-" * 110)
    i = 0
    while i < max(len(cmp["plain"]), len(cmp["boosted"])):
        if i < len(cmp["plain"]):
            ld, ls = cmp["plain"][i]
            left = "%-7s %-9.4f %-30s" % (ld, ls, eng.index.title_of(ld)[:30])
        else:
            left = " " * 48
        if i < len(cmp["boosted"]):
            r = cmp["boosted"][i]
            if r["span"] > 0:
                span = str(r["span"])
            else:
                span = "-"
            right = "%-7s %-9.4f %-5s %s" % (r["docid"], r["score"], span,
                                             eng.index.title_of(r["docid"]))
        else:
            right = ""
        print("   %-4d %s  %s" % (i + 1, left, right))
        i = i + 1

    print()
    if cmp["order_changed"]:
        print("   the two rankings are DIFFERENT.")
        if len(cmp["entered"]) > 0:
            print("   entered the top 10 because of the positions : %s"
                  % ", ".join(cmp["entered"]))
        if len(cmp["left"]) > 0:
            print("   dropped out of the top 10                   : %s"
                  % ", ".join(cmp["left"]))
    else:
        print("   both rankings are identical for this query - the query terms "
              "are already")
        print("   close together in every document that matched.")


def print_term(eng, word):
    import preprocess
    pairs = preprocess.analyse(word)
    if len(pairs) == 0:
        print("   \"%s\" is a stop word, it is not in the index." % word)
        return
    term = pairs[0][1]
    print()
    print('   "%s"  stems to  "%s"' % (word, term))
    if term not in eng.index.df:
        print("   this term is not in the dictionary.")
        import novelty
        s, d = novelty.suggest_term(eng.index, term)
        if s is not None:
            print('   closest term we do have: "%s" (edit distance %d)' % (s, d))
        return
    print("   df  = %d" % eng.index.df[term])
    print("   idf = log10(%d/%d) = %.4f"
          % (eng.index.N, eng.index.df[term], eng.index.idf(term)))
    print()
    print("   inverted index postings (docID, tf):")
    line = []
    for docid, tf in eng.index.postings(term)[:20]:
        line.append("(%s,%d)" % (docid, tf))
    print("     " + ", ".join(line))
    if eng.index.df[term] > 20:
        print("     ... %d more" % (eng.index.df[term] - 20))
    print()
    print("   positional index postings (docID, tf, [positions]):")
    for docid, tf, plist in eng.index.positional_postings(term)[:8]:
        positions = []
        for p in plist:
            positions.append(str(p))
        print("     (%s, tf=%d, [%s])" % (docid, tf, ", ".join(positions)))
    if eng.index.df[term] > 8:
        print("     ... %d more" % (eng.index.df[term] - 8))


def main():
    print(BANNER)
    print("loading the corpus and building the indexes ...")
    eng = engine.ClothingSearchEngine()
    st = eng.stats()
    print("  %d documents, %d distinct terms, %d postings"
          % (st["documents"], st["terms"], st["postings"]))
    print(HELP)

    mode = "free"
    boost_on = True
    last_query = ""

    while True:
        try:
            raw = input("[%s] > " % mode).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw == "":
            continue

        lower = raw.lower()

        # --- commands ---------------------------------------------------
        if lower in ("quit", "exit", "q"):
            break

        if lower in ("help", "?"):
            print(HELP)
            continue

        if lower in ("1", "free", "mode 1"):
            mode = "free"
            print("   free text mode. Type a query.")
            continue

        if lower in ("2", "phrase", "positional", "mode 2"):
            mode = "phrase"
            print("   phrase / proximity mode. Type a phrase, or "
                  "term1 WITHIN/k term2")
            continue

        if lower in ("3", "compare", "mode 3"):
            mode = "compare"
            print("   comparison mode. Type a query to see both rankings.")
            continue

        if lower == "stats":
            print()
            print("   documents (N)        : %d" % st["documents"])
            print("   dictionary terms     : %d" % st["terms"])
            print("   postings             : %d" % st["postings"])
            print("   proximity boost      : %s (alpha = %.2f)"
                  % ("on" if boost_on else "off", eng.alpha))
            zero = 0
            for t in eng.index.df:
                if eng.index.df[t] == eng.index.N:
                    zero = zero + 1
            print("   terms with idf = 0   : %d   (they appear in every "
                  "document)" % zero)
            continue

        if lower.startswith("term "):
            print_term(eng, raw[5:].strip())
            continue

        if lower.startswith("explain "):
            docid = raw[8:].strip().upper()
            if last_query == "":
                print("   run a free text query first.")
            elif docid not in eng.index.docs:
                print("   no such document: %s" % docid)
            else:
                print()
                print(eng.explain(last_query, docid))
            continue

        if lower in ("boost on", "boost off"):
            boost_on = lower.endswith("on")
            print("   proximity boost is now %s" % ("on" if boost_on else "off"))
            continue

        if lower.startswith("alpha "):
            try:
                eng.alpha = float(raw.split()[1])
                print("   alpha = %.2f" % eng.alpha)
            except (ValueError, IndexError):
                print("   usage: alpha 0.5")
            continue

        # --- an actual query --------------------------------------------
        if mode == "free":
            last_query = raw
            out = eng.search(raw, 10, use_boost=boost_on, autocorrect=True)
            print_free_text(out, boost_on)
        elif mode == "phrase":
            out = eng.run_positional_query(raw, 10)
            print_positional(out)
        else:
            last_query = raw
            print_compare(eng, raw)

        print()

    print("bye.")


if __name__ == "__main__":
    main()
