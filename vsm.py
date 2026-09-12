# Part B: ranked retrieval with lnc.ltc.
#
#   documents "lnc" : 1 + log10(tf), no idf, cosine normalised
#   query     "ltc" : (1 + log10(tf)) * log10(N/df), cosine normalised
#
# Document weights are already divided by the stored document length, so
# the cosine similarity is just the dot product of the two vectors.

import math

import preprocess


def query_weights(index, query_string):
    """Build the normalised query vector. Returns (weights, info)."""

    pairs = preprocess.analyse(query_string)

    raw_tf = {}
    order = []
    for position, term in pairs:
        if term not in raw_tf:
            raw_tf[term] = 0
            order.append(term)
        raw_tf[term] = raw_tf[term] + 1

    info = {
        "terms": order,
        "unknown": [],      # not in the dictionary
        "zero_idf": [],     # df == N, so idf == 0
        "fallback": False,
    }

    weights = {}
    for term in order:
        if term not in index.df:
            info["unknown"].append(term)
            continue
        idf = index.idf(term)
        if idf == 0.0:
            info["zero_idf"].append(term)
        weights[term] = (1.0 + math.log10(raw_tf[term])) * idf

    total = 0.0
    for term in weights:
        total = total + weights[term] * weights[term]
    length = math.sqrt(total)

    if length == 0.0:
        # Happens on this corpus because all 100 products end with the same
        # boilerplate sentence, so wear / regular / comfort / garment have
        # df = 100 and idf = 0. A query built only from those gives a zero
        # vector and nothing can be ranked. Drop the idf factor (lnc.lnc)
        # and flag it so the interface can say what happened.
        info["fallback"] = True
        for term in order:
            if term in index.df:
                weights[term] = 1.0 + math.log10(raw_tf[term])
        total = 0.0
        for term in weights:
            total = total + weights[term] * weights[term]
        length = math.sqrt(total)

    if length > 0.0:
        for term in weights:
            weights[term] = weights[term] / length

    return weights, info


def cosine_scores(index, query_string):
    """Term at a time accumulation over the query terms' postings lists.
    Documents with no query term score 0 and are never visited."""

    weights, info = query_weights(index, query_string)

    scores = {}
    for term in weights:
        wq = weights[term]
        if wq == 0.0:
            continue
        for docid, tf in index.postings(term):
            wd = 1.0 + math.log10(tf)
            if docid not in scores:
                scores[docid] = 0.0
            scores[docid] = scores[docid] + (wq * wd)

    for docid in scores:
        length = index.doc_length[docid]
        if length > 0.0:
            scores[docid] = scores[docid] / length

    # count before the list is cut to 10
    info["matched_docs"] = len(scores)

    return scores, info


def rank(index, query_string, top_k=10):
    """Top k as (docID, score), decreasing score, ties by increasing docID."""
    scores, info = cosine_scores(index, query_string)

    ranked = []
    for docid in scores:
        ranked.append((docid, scores[docid]))

    # negate the score so tuple sorting gives decreasing score but the
    # docID still sorts ascending
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))

    return ranked[:top_k], info


def explain(index, query_string, docid):
    """Per term arithmetic for one document, used by the CLI and the GUI."""

    weights, info = query_weights(index, query_string)
    lines = []
    lines.append("score breakdown for %s  (query: %s)" % (docid, query_string))
    lines.append("%-14s %6s %8s %10s %10s %10s"
                 % ("term", "tf_d", "df", "w_query", "w_doc", "product"))
    lines.append("-" * 64)

    total = 0.0
    for term in info["terms"]:
        if term not in weights:
            lines.append("%-14s %6s %8s %10s %10s %10s"
                         % (term, "-", "-", "not in dictionary", "", ""))
            continue
        tf = index.term_frequency(term, docid)
        if tf == 0:
            wd = 0.0
        else:
            wd = 1.0 + math.log10(tf)
        wq = weights[term]
        product = wq * wd
        total = total + product
        lines.append("%-14s %6d %8d %10.4f %10.4f %10.4f"
                     % (term, tf, index.df.get(term, 0), wq, wd, product))

    lines.append("-" * 64)
    lines.append("dot product                                        %10.4f" % total)
    lines.append("document length (lnc)                              %10.4f"
                 % index.doc_length[docid])
    if index.doc_length[docid] > 0:
        lines.append("cosine score = dot / length                        %10.4f"
                     % (total / index.doc_length[docid]))
    return "\n".join(lines)


if __name__ == "__main__":
    import indexer
    idx = indexer.build_index("data/corpus_100.txt")
    results, info = rank(idx, "black cotton t-shirt")
    for docid, score in results:
        print("%s  %.4f  %s" % (docid, score, idx.title_of(docid)))
    print()
    print(explain(idx, "black cotton t-shirt", results[0][0]))
