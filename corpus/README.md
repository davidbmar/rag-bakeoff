# Corpus

The PDFs are not committed — they are large, and they are freely available from
the source. Fetch them with:

```bash
for pub in p527 p925 p946 p523; do
  curl -sL -o "$pub.pdf" "https://www.irs.gov/pub/irs-pdf/$pub.pdf"
done
curl -sL -o i1040gi.pdf https://www.irs.gov/pub/irs-pdf/i1040gi.pdf
```

| file | publication |
|---|---|
| `p527` | Residential Rental Property |
| `p925` | Passive Activity and At-Risk Rules |
| `p946` | How To Depreciate Property |
| `p523` | Selling Your Home |
| `i1040gi` | Instructions for Form 1040 |

Extracted `.txt` alongside each PDF is what the keyword engine and the
full-context control read; the semantic engine is handed the PDF so that its
own parser is part of what gets compared. Any PDF text extractor will do —
`pypdf` and `PyMuPDF` were both used here.

US federal government works, public domain.
