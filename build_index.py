# Run this first:  python build_index.py
# Builds both indexes from data/corpus_100.txt and writes the dumps in
# output/ (dictionary, inverted index, positional index, doc lengths).

import os
import time

import indexer

CORPUS = os.path.join("data", "corpus_100.txt")
OUTDIR = "output"


def main():
    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)

    print("reading", CORPUS, "...")
    start = time.time()

    idx = indexer.SearchIndex()
    n = idx.load_corpus(CORPUS)
    print("  %d documents read" % n)

    idx.build()
    elapsed = time.time() - start

    print("  %d distinct terms in the dictionary" % len(idx.inverted))
    total_postings = 0
    for term in idx.inverted:
        total_postings = total_postings + len(idx.inverted[term])
    print("  %d postings in total" % total_postings)

    total_tokens = 0
    for docid in idx.doc_tokens:
        total_tokens = total_tokens + idx.doc_tokens[docid]
    print("  %d terms indexed after stop word removal" % total_tokens)
    print("  built in %.3f seconds" % elapsed)

    print()
    print("writing the index dumps ...")

    idx.dump_inverted_index(os.path.join(OUTDIR, "inverted_index.txt"))
    print("   output/inverted_index.txt")

    idx.dump_positional_index(os.path.join(OUTDIR, "positional_index.txt"))
    print("   output/positional_index.txt")

    idx.dump_dictionary(os.path.join(OUTDIR, "dictionary.txt"))
    print("   output/dictionary.txt")

    idx.dump_doc_lengths(os.path.join(OUTDIR, "doc_lengths.txt"))
    print("   output/doc_lengths.txt")

    # collection summary, the numbers quoted in the report
    import preprocess
    f = open(os.path.join(OUTDIR, "corpus_stats.txt"), "w", encoding="utf-8")
    f.write("CORPUS STATISTICS\n")
    f.write("=" * 60 + "\n")
    f.write("documents (N)                 : %d\n" % idx.N)
    f.write("distinct terms (dictionary)   : %d\n" % len(idx.inverted))
    f.write("postings                      : %d\n" % total_postings)
    f.write("terms after preprocessing     : %d\n" % total_tokens)
    f.write("average terms per document    : %.1f\n"
            % (float(total_tokens) / idx.N))
    f.write("stop words in our list        : %d\n" % len(preprocess.STOPWORDS))
    f.write("words taken back OUT of the standard list : %s\n"
            % ", ".join(preprocess.KEEP_THESE))
    f.write("\n")
    f.write("Terms with df = N (they appear in every single document, so\n")
    f.write("idf = log10(100/100) = 0 and they cannot affect the cosine\n")
    f.write("score at all). This is the reason we built the proximity boost:\n")
    f.write("-" * 60 + "\n")
    zero = []
    for term in idx.vocabulary():
        if idx.df[term] == idx.N:
            zero.append(term)
    for term in zero:
        f.write("   %s\n" % term)
    f.write("\n%d of the %d dictionary terms are in this state.\n"
            % (len(zero), len(idx.inverted)))

    f.write("\n\nCATEGORY BREAKDOWN\n")
    f.write("-" * 60 + "\n")
    counts = {}
    for docid in idx.doc_order:
        c = idx.category_of(docid)
        counts[c] = counts.get(c, 0) + 1
    for c in sorted(counts):
        f.write("   %-14s %d\n" % (c, counts[c]))
    f.close()
    print("   output/corpus_stats.txt")

    print()
    print("done. now run:  python run_tests.py     (Part E)")
    print("            or: python search_cli.py    (Part D, terminal)")
    print("            or: python search_gui.py    (Part D, window)")


if __name__ == "__main__":
    main()
