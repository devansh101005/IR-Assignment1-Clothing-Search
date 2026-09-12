# Part D, window interface:  python search_gui.py
# tkinter, so nothing has to be installed. The four radio buttons are
# free text (Part B), exact phrase (Part C), proximity WITHIN/k (Part C)
# and a side by side comparison. The panel at the bottom shows the
# positions and the score arithmetic behind whatever was just run.

import tkinter as tk
from tkinter import ttk

import engine
import novelty
import preprocess


BG = "#f4f4f2"
HEAD = "#2b3a55"


class SearchApp(object):

    def __init__(self, root):
        self.root = root
        self.root.title("Clothing Search Engine - CSD358 Assignment 1")
        self.root.geometry("1180x800")
        self.root.configure(bg=BG)

        self.engine = engine.ClothingSearchEngine()
        self.stats = self.engine.stats()
        self.last_query = ""
        self.columns = []

        self._build_header()
        self._build_controls()
        self._build_results()
        self._build_evidence()
        self._build_status()

        self.entry.focus_set()
        self.on_mode_change()

    # ------------------------------------------------------------------
    def _build_header(self):
        bar = tk.Frame(self.root, bg=HEAD)
        bar.pack(fill="x")

        tk.Label(bar, text="   Clothing Search Engine",
                 bg=HEAD, fg="white",
                 font=("Segoe UI", 17, "bold")).pack(side="left", pady=10)

        tk.Label(bar,
                 text="CSD358 Information Retrieval\nAssignment 1   ",
                 bg=HEAD, fg="#c8d4e8", justify="right",
                 font=("Segoe UI", 9)).pack(side="right", pady=8, padx=16)

    # ------------------------------------------------------------------
    def _build_controls(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="x", padx=12, pady=(12, 4))

        # --- the search box ---
        row = tk.Frame(outer, bg=BG)
        row.pack(fill="x")

        tk.Label(row, text="Query:", bg=BG,
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        self.query_var = tk.StringVar()
        self.entry = tk.Entry(row, textvariable=self.query_var,
                              font=("Consolas", 13), relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True, padx=8, ipady=5)
        self.entry.bind("<Return>", self.on_search)

        tk.Button(row, text="Search", command=self.on_search,
                  font=("Segoe UI", 10, "bold"), bg=HEAD, fg="white",
                  relief="flat", padx=18, pady=4).pack(side="left")
        tk.Button(row, text="Clear", command=self.on_clear,
                  font=("Segoe UI", 10), relief="flat",
                  padx=12, pady=4).pack(side="left", padx=(6, 0))

        # --- the mode selector and the options ---
        row2 = tk.Frame(outer, bg=BG)
        row2.pack(fill="x", pady=(8, 0))

        self.mode = tk.StringVar(value="free")
        modes = [
            ("Free text (ranked, Part B)", "free"),
            ("Exact phrase (Part C)", "phrase"),
            ("Proximity WITHIN/k (Part C)", "prox"),
            ("Compare with / without positions", "compare"),
        ]
        for label, value in modes:
            tk.Radiobutton(row2, text=label, variable=self.mode, value=value,
                           bg=BG, font=("Segoe UI", 9),
                           command=self.on_mode_change).pack(side="left",
                                                             padx=(0, 12))

        tk.Label(row2, text="k =", bg=BG,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 2))
        self.k_var = tk.IntVar(value=3)
        self.k_spin = tk.Spinbox(row2, from_=1, to=50, width=4,
                                 textvariable=self.k_var,
                                 font=("Segoe UI", 9))
        self.k_spin.pack(side="left")

        row3 = tk.Frame(outer, bg=BG)
        row3.pack(fill="x", pady=(6, 0))

        self.boost_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row3, text="use proximity boost",
                       variable=self.boost_var, bg=BG,
                       font=("Segoe UI", 9)).pack(side="left")

        self.correct_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row3, text="correct spelling mistakes",
                       variable=self.correct_var, bg=BG,
                       font=("Segoe UI", 9)).pack(side="left", padx=(14, 0))

        tk.Label(row3, text="alpha", bg=BG,
                 font=("Segoe UI", 9)).pack(side="left", padx=(16, 4))
        self.alpha_var = tk.DoubleVar(value=0.5)
        tk.Scale(row3, from_=0.0, to=2.0, resolution=0.1, orient="horizontal",
                 variable=self.alpha_var, bg=BG, length=140,
                 showvalue=True, font=("Segoe UI", 8)).pack(side="left")

        # shortcuts so a demo does not need typing
        row4 = tk.Frame(outer, bg=BG)
        row4.pack(fill="x", pady=(4, 0))
        tk.Label(row4, text="try:", bg=BG, fg="#555",
                 font=("Segoe UI", 9)).pack(side="left")
        samples = ["black cotton t-shirt", "regular fit", "cotton shirt",
                   "stretch denim", "winter wear", "high waist",
                   "cotton WITHIN/3 shirt", "blak hoodei"]
        for s in samples:
            tk.Button(row4, text=s, font=("Segoe UI", 8), relief="groove",
                      bg="#e8e8e4", padx=5,
                      command=lambda t=s: self.on_sample(t)).pack(side="left",
                                                                  padx=2)

    # ------------------------------------------------------------------
    def _build_results(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="both", expand=True, padx=12, pady=(10, 4))

        self.result_label = tk.Label(frame, text="", bg=BG, anchor="w",
                                     font=("Segoe UI", 10, "bold"))
        self.result_label.pack(fill="x")

        holder = tk.Frame(frame)
        holder.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("Treeview", font=("Consolas", 10), rowheight=23)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        self.tree = ttk.Treeview(holder, show="headings", height=12)
        scroll = ttk.Scrollbar(holder, orient="vertical",
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.tree.bind("<Double-1>", self.on_row_double_click)
        self.tree.tag_configure("odd", background="#f7f7fb")

    # ------------------------------------------------------------------
    def _build_evidence(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="both", padx=12, pady=(2, 4))

        tk.Label(frame, text="Details / evidence from the index"
                             "   (double click a result row for the full "
                             "lnc.ltc calculation)",
                 bg=BG, anchor="w", fg="#333",
                 font=("Segoe UI", 9, "bold")).pack(fill="x")

        holder = tk.Frame(frame)
        holder.pack(fill="both")

        self.evidence = tk.Text(holder, height=11, font=("Consolas", 9),
                                wrap="none", relief="solid", bd=1,
                                bg="#fbfbf9")
        vs = ttk.Scrollbar(holder, orient="vertical",
                           command=self.evidence.yview)
        hs = ttk.Scrollbar(frame, orient="horizontal",
                           command=self.evidence.xview)
        self.evidence.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.evidence.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        hs.pack(fill="x")

    # ------------------------------------------------------------------
    def _build_status(self):
        self.status = tk.Label(
            self.root, anchor="w", bg="#e4e4e0", fg="#333",
            font=("Segoe UI", 9),
            text="  %d documents indexed   |   %d distinct terms   |   "
                 "%d postings   |   weighting lnc.ltc   |   ready"
                 % (self.stats["documents"], self.stats["terms"],
                    self.stats["postings"]))
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def set_columns(self, spec):
        """spec is a list of (heading, width, anchor)."""
        self.tree.delete(*self.tree.get_children())
        names = []
        for i in range(len(spec)):
            names.append("c%d" % i)
        self.tree["columns"] = names
        i = 0
        for heading, width, anchor in spec:
            self.tree.heading(names[i], text=heading)
            self.tree.column(names[i], width=width, anchor=anchor,
                             stretch=(i == len(spec) - 1))
            i = i + 1
        self.columns = spec

    def add_row(self, values, n):
        tag = "odd" if n % 2 else ""
        self.tree.insert("", "end", values=values, tags=(tag,))

    def say(self, text):
        self.evidence.delete("1.0", "end")
        self.evidence.insert("1.0", text)

    def on_clear(self):
        self.query_var.set("")
        self.tree.delete(*self.tree.get_children())
        self.evidence.delete("1.0", "end")
        self.result_label.config(text="")
        self.entry.focus_set()

    def on_sample(self, text):
        self.query_var.set(text)
        if "WITHIN" in text:
            self.mode.set("prox")
        self.on_mode_change()
        self.on_search()

    def on_mode_change(self):
        mode = self.mode.get()
        if mode == "prox":
            self.k_spin.configure(state="normal")
        else:
            self.k_spin.configure(state="disabled")

    # ------------------------------------------------------------------
    # the search button
    # ------------------------------------------------------------------
    def on_search(self, event=None):
        query = self.query_var.get().strip()
        if query == "":
            return

        self.engine.alpha = self.alpha_var.get()
        mode = self.mode.get()

        if mode == "free":
            self.run_free_text(query)
        elif mode == "phrase":
            self.run_phrase(query)
        elif mode == "prox":
            self.run_proximity(query)
        else:
            self.run_compare(query)

    # ------------------------------------------------------------------
    def run_free_text(self, query):
        self.last_query = query
        boost = self.boost_var.get()
        out = self.engine.search(query, 10, use_boost=boost,
                                 autocorrect=self.correct_var.get())
        self.last_query = out["used_query"]

        if boost:
            self.set_columns([("Rank", 55, "center"), ("DocID", 70, "center"),
                              ("Score", 85, "center"), ("Cosine", 85, "center"),
                              ("Span", 55, "center"),
                              ("Category", 100, "w"),
                              ("Product title", 420, "w")])
        else:
            self.set_columns([("Rank", 55, "center"), ("DocID", 70, "center"),
                              ("Cosine score", 110, "center"),
                              ("Category", 110, "w"),
                              ("Product title", 480, "w")])

        n = 1
        for r in out["results"]:
            if boost:
                span = str(r["span"]) if r["span"] > 0 else "-"
                self.add_row((n, r["docid"], "%.4f" % r["score"],
                              "%.4f" % r["base"], span, r["category"],
                              r["title"]), n)
            else:
                self.add_row((n, r["docid"], "%.4f" % r["score"],
                              r["category"], r["title"]), n)
            n = n + 1

        self.result_label.config(
            text="Top %d of %d matching documents   -   %s"
                 % (len(out["results"]), out["total_matched"],
                    "lnc.ltc cosine with proximity boost" if boost
                    else "plain lnc.ltc cosine (Part B)"))

        # --- the evidence panel ---
        lines = []
        lines.append('query as typed          : "%s"' % out["original_query"])
        if len(out["corrections"]) > 0:
            for typed, fixed in out["corrections"]:
                lines.append('  spelling              : "%s" is not in the '
                             'dictionary, using "%s"' % (typed, fixed))
            lines.append('query actually used     : "%s"' % out["used_query"])
        if len(out["not_found"]) > 0:
            lines.append("  ignored (not in corpus): %s"
                         % ", ".join(out["not_found"]))
        lines.append("terms after Part A pipeline : %s" % out["stems"])

        lines.append("")
        lines.append("%-14s %6s %10s %12s" % ("term", "df", "idf", "in dict?"))
        lines.append("-" * 48)
        for t in out["stems"]:
            if t in self.engine.index.df:
                lines.append("%-14s %6d %10.4f %12s"
                             % (t, self.engine.index.df[t],
                                self.engine.index.idf(t), "yes"))
            else:
                lines.append("%-14s %6s %10s %12s" % (t, "-", "-", "NO"))

        if len(out["zero_idf_terms"]) > 0:
            lines.append("")
            lines.append("terms with idf = 0 (df = N = 100): %s"
                         % ", ".join(out["zero_idf_terms"]))
            lines.append("these words appear in every single product, so they "
                         "carry no weight in the cosine score.")
            lines.append("The proximity boost can still use their POSITIONS.")
        if out["fallback"]:
            lines.append("")
            lines.append("*** every query term has idf = 0, so the plain "
                         "vector space model cannot rank this query. ***")
            lines.append("We dropped the idf factor for this query, and the "
                         "positions are doing the real work.")

        if boost and len(out["results"]) > 0:
            lines.append("")
            lines.append("proximity boost (score = cosine x (1 + "
                         "%.2f x coverage x tightness))" % self.engine.alpha)
            lines.append("%-8s %9s %9s %6s %9s %11s %9s"
                         % ("docID", "cosine", "final", "span", "matched",
                            "tightness", "boost"))
            lines.append("-" * 72)
            for r in out["results"]:
                lines.append("%-8s %9.4f %9.4f %6s %9s %11.3f %9.3f"
                             % (r["docid"], r["base"], r["score"],
                                (str(r["span"]) if r["span"] > 0 else "-"),
                                "%d/%d" % (r["matched"], r["asked"]),
                                (float(r["matched"]) / r["span"]
                                 if r["span"] > 0 else 0.0),
                                r["boost"]))

        self.say("\n".join(lines))
        self.set_status("free text search finished, %d documents matched"
                        % out["total_matched"])

    # ------------------------------------------------------------------
    def run_phrase(self, query):
        out = self.engine.phrase(query, 10)

        self.set_columns([("Rank", 55, "center"), ("DocID", 70, "center"),
                          ("Occurrences", 95, "center"),
                          ("Positions in document", 190, "w"),
                          ("Category", 100, "w"),
                          ("Product title", 420, "w")])

        n = 1
        for r in out["results"]:
            plist = []
            for p in r["positions"]:
                plist.append(str(p))
            self.add_row((n, r["docid"], r["occurrences"],
                          "[" + ", ".join(plist) + "]", r["category"],
                          r["title"]), n)
            n = n + 1

        self.result_label.config(
            text='Exact phrase "%s"  -  %d document(s) contain it'
                 % (query, out["total"]))

        lines = []
        lines.append('phrase as typed           : "%s"' % out["phrase"])
        lines.append("terms after Part A pipeline: %s" % out["stems"])
        lines.append("documents containing the exact phrase: %d" % out["total"])
        lines.append("")

        if out["total"] == 0:
            lines.append("Nothing found. That is NOT the same as the words "
                         "being missing from the collection:")
            for t in out["stems"]:
                if t in self.engine.index.df:
                    lines.append('   "%s" IS in the dictionary with df = %d'
                                 % (t, self.engine.index.df[t]))
                else:
                    lines.append('   "%s" is not in the dictionary at all' % t)
            lines.append("")
            lines.append("If the words are present but the phrase is not, it "
                         "means no document ever puts")
            lines.append("them next to each other. A plain vector space search "
                         "would still have returned")
            lines.append("documents here, which is exactly why the positional "
                         "index is needed.")
        else:
            lines.append("EVIDENCE - these positions come straight out of the "
                         "positional index,")
            lines.append("the phrase was confirmed by checking that the "
                         "positions are consecutive:")
            lines.append("")
            for r in out["results"]:
                lines.append("%s  positions %s" % (r["docid"],
                                                   str(r["positions"])))
                lines.append("      %s" % r["snippet"])
            lines.append("")
            lines.append("postings actually consulted:")
            for t in out["stems"]:
                pp = self.engine.index.positional_postings(t)[:6]
                shown = []
                for docid, tf, plist in pp:
                    shown.append("(%s, tf=%d, %s)" % (docid, tf, plist))
                lines.append("   %-12s df=%-4d %s%s"
                             % (t, self.engine.index.df.get(t, 0),
                                "  ".join(shown),
                                "  ..." if self.engine.index.df.get(t, 0) > 6
                                else ""))

        self.say("\n".join(lines))
        self.set_status("phrase search finished, %d documents" % out["total"])

    # ------------------------------------------------------------------
    def run_proximity(self, query):
        # accepts "cotton WITHIN/3 shirt" or two plain words plus the
        # k from the spin box
        kind, args = self.engine.parse_positional_query(query)
        if kind == "proximity":
            t1, t2, k = args
            self.k_var.set(k)
        else:
            words = query.replace('"', "").split()
            if len(words) < 2:
                self.say("A proximity query needs two terms, for example:\n\n"
                         "    cotton shirt          (with k from the spin box)\n"
                         "    cotton WITHIN/3 shirt (k written in the query)")
                return
            t1 = words[0]
            t2 = words[-1]
            k = self.k_var.get()

        out = self.engine.proximity(t1, t2, k, True, 10)

        self.set_columns([("Rank", 55, "center"), ("DocID", 70, "center"),
                          ("Closest gap", 95, "center"),
                          ("Matching (p1, p2) pairs", 230, "w"),
                          ("Category", 100, "w"),
                          ("Product title", 400, "w")])

        n = 1
        for r in out["results"]:
            pairs = []
            for a, b in r["pairs"][:4]:
                pairs.append("(%d,%d)" % (a, b))
            tail = " ..." if len(r["pairs"]) > 4 else ""
            self.add_row((n, r["docid"], r["closest"],
                          ", ".join(pairs) + tail, r["category"],
                          r["title"]), n)
            n = n + 1

        self.result_label.config(
            text="%s WITHIN/%d %s  -  %d document(s), ordered proximity"
                 % (t1, k, t2, out["total"]))

        lines = []
        lines.append("ordered proximity query : %s WITHIN/%d %s" % (t1, k, t2))
        lines.append("terms after Part A pipeline : %s" % str(out["stems"]))
        lines.append("a document matches when  0 < p2 - p1 <= %d" % k)
        lines.append("matching documents : %d" % out["total"])
        lines.append("")

        if out["total"] == 0:
            lines.append("No document has these two terms that close together "
                         "in this order.")
            lines.append("Try increasing k, or swapping the two terms - the "
                         "order matters.")
        else:
            lines.append("EVIDENCE - the (p1, p2) pairs below are read from "
                         "the positional index:")
            lines.append("")
            for r in out["results"][:6]:
                lines.append("%s  closest gap %d  pairs %s"
                             % (r["docid"], r["closest"], str(r["pairs"][:5])))
                lines.append("      %s" % r["snippet"])

            lines.append("")
            lines.append("how the answer changes with k:")
            lines.append("%-6s %s" % ("k", "documents"))
            for kk in [1, 2, 3, 4, 5, 8, 12, 20]:
                sweep = self.engine.proximity(t1, t2, kk, True, 100)
                mark = "   <-- current" if kk == k else ""
                lines.append("%-6d %d%s" % (kk, sweep["total"], mark))

        self.say("\n".join(lines))
        self.set_status("proximity search finished, %d documents"
                        % out["total"])

    # ------------------------------------------------------------------
    def run_compare(self, query):
        self.last_query = query
        self.engine.alpha = self.alpha_var.get()
        cmp = self.engine.compare(query, 10)

        self.set_columns([("Rank", 50, "center"),
                          ("DocID", 65, "center"),
                          ("Cosine", 80, "center"),
                          ("Title - plain VSM (Part B)", 330, "w"),
                          ("DocID", 65, "center"),
                          ("Score", 80, "center"),
                          ("Span", 50, "center"),
                          ("Title - with positions", 330, "w")])

        n = 1
        i = 0
        while i < max(len(cmp["plain"]), len(cmp["boosted"])):
            if i < len(cmp["plain"]):
                ld, ls = cmp["plain"][i]
                left = (ld, "%.4f" % ls, self.engine.index.title_of(ld))
            else:
                left = ("", "", "")
            if i < len(cmp["boosted"]):
                r = cmp["boosted"][i]
                span = str(r["span"]) if r["span"] > 0 else "-"
                right = (r["docid"], "%.4f" % r["score"], span,
                         self.engine.index.title_of(r["docid"]))
            else:
                right = ("", "", "", "")
            self.add_row((n,) + left + right, n)
            n = n + 1
            i = i + 1

        if cmp["order_changed"]:
            verdict = "the two rankings are DIFFERENT"
        else:
            verdict = "both rankings are identical for this query"
        self.result_label.config(
            text='Comparison for "%s"  -  %s' % (query, verdict))

        lines = []
        lines.append('query : "%s"' % query)
        lines.append("left  : plain lnc.ltc from Part B, positions ignored")
        lines.append("right : the same cosine multiplied by "
                     "(1 + %.2f x coverage x tightness)" % self.engine.alpha)
        lines.append("")
        lines.append("order changed                       : %s"
                     % cmp["order_changed"])
        lines.append("entered the top 10 due to positions : %s"
                     % (", ".join(cmp["entered"])
                        if len(cmp["entered"]) > 0 else "none"))
        lines.append("dropped out of the top 10           : %s"
                     % (", ".join(cmp["left"])
                        if len(cmp["left"]) > 0 else "none"))
        lines.append("")

        zero = cmp["info"]["zero_idf"]
        if len(zero) > 0:
            lines.append("terms in this query with idf = 0 : %s"
                         % ", ".join(zero))
            lines.append("The cosine score cannot see these words at all. "
                         "The positional index can,")
            lines.append("and that is where the difference above comes from.")
            lines.append("")

        lines.append("%-8s %9s %9s %6s %9s %9s   %s"
                     % ("docID", "cosine", "final", "span", "matched",
                        "boost", "title"))
        lines.append("-" * 100)
        for r in cmp["boosted"]:
            lines.append("%-8s %9.4f %9.4f %6s %9s %9.3f   %s"
                         % (r["docid"], r["base"], r["score"],
                            (str(r["span"]) if r["span"] > 0 else "-"),
                            "%d/%d" % (r["matched"], r["asked"]),
                            r["boost"],
                            self.engine.index.title_of(r["docid"])))

        self.say("\n".join(lines))
        self.set_status("comparison finished")

    # ------------------------------------------------------------------
    def on_row_double_click(self, event):
        selection = self.tree.selection()
        if len(selection) == 0:
            return
        values = self.tree.item(selection[0])["values"]

        # docID is the second column in every layout
        docid = None
        for v in values:
            text = str(v).strip().upper()
            if text in self.engine.index.docs:
                docid = text
                break
        if docid is None:
            return

        d = self.engine.index.docs[docid]
        lines = []
        lines.append("%s   %s" % (docid, d["title"]))
        lines.append("category : %s" % d["category"])
        lines.append("-" * 100)
        lines.append(d["text"])
        lines.append("")

        if self.last_query != "":
            lines.append(self.engine.explain(self.last_query, docid))
            lines.append("")

        # show where each query term sits inside this document
        terms = []
        for position, term in preprocess.analyse(self.last_query):
            if term not in terms:
                terms.append(term)
        if len(terms) > 0:
            lines.append("positions of the query terms inside %s:" % docid)
            lists = []
            for t in terms:
                plist = self.engine.index.positional.get(t, {}).get(docid)
                if plist is None:
                    lines.append("   %-12s not present in this document" % t)
                else:
                    lines.append("   %-12s %s" % (t, plist))
                    lists.append(plist)
            if len(lists) >= 2:
                lines.append("   shortest window covering them : %d tokens"
                             % novelty.smallest_window(lists))

        self.say("\n".join(lines))
        self.set_status("showing document %s" % docid)

    # ------------------------------------------------------------------
    def set_status(self, message):
        self.status.config(
            text="  %d documents   |   %d terms   |   %d postings   |   %s"
                 % (self.stats["documents"], self.stats["terms"],
                    self.stats["postings"], message))


def main():
    root = tk.Tk()
    SearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
