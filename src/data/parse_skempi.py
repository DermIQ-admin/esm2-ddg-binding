"""Parse SKEMPI mutation strings and compute the ddG regression target.

Implements PLAN.md sections 6.2-6.4.

TODO(session 2): implement.

Ordering matters here — section 6.2 is explicit that we inspect the real file
before coding against an assumed schema. Column names and the mutation-string
format both get confirmed against the downloaded CSV first.

  1. Inspect: print df.columns.tolist() and df.head() before writing parsers
  2. Parse mutation strings (roughly `<wt_aa><chain><position><mut_aa>`,
     comma-separated for multi-point entries)
  3. Compute ddG (section 6.3):
         ddG = dG(mutant) - dG(wild-type),  dG = R * T * ln(Kd)
         R = 1.987e-3 kcal/(mol*K), T from SKEMPI's Temperature column
         where available, else 298.15 K
     Sign convention: POSITIVE = destabilizing (weaker binding).
     Sanity check from the plan: Kd_mut = 1 uM against Kd_wt = 1 nM
     (1000x weaker) must give ddG ~ +4.09 kcal/mol.
  4. Filter (section 6.4):
     - single-point mutations only for v1 (drop comma-separated entries)
     - drop the ~440 "abolishes binding" entries: they are inequalities /
       lower bounds, not clean Kd values. Do NOT invent a number for them.
       Flag them separately in case they become a classification signal later.
  5. Build the WT and mutant sequences for the concatenated complex, and
     verify they are the same length (section 14 pitfall: SKEMPI mutations
     are overwhelmingly substitutions, but verify rather than assume)
"""
