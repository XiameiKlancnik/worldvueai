import pandas as pd
from pathlib import Path

path = Path('articles_with_embeddings.parquet')
df = pd.read_parquet(path)
initial = len(df)

# Language filter: keep English only
non_en = (df['language'].astype(str).str.lower() != 'en').sum() if 'language' in df.columns else 0

# Exclude keywords
exclude_terms = [t.strip().lower() for t in 'celebrity,entertainment,sports,lifestyle,hollywood,showbiz,olympics,tennis,football,soccer,basketball,cricket,fashion,beauty,gossip,streaming,series,netflix,hbo,concert,festival'.split(',') if t.strip()]

cols = [c for c in ['keywords','frames','category','section','topic','title','summary','text'] if c in df.columns]
if cols:
    txt = df[cols].fillna('').astype(str)
    haystack = txt.agg(' '.join, axis=1).str.lower()
    kw_mask = haystack.apply(lambda t: any(term in t for term in exclude_terms))
    kw_excluded = int(kw_mask.sum())
else:
    kw_excluded = 0

# Combine masks (note: some rows may be both non-en and keyword hits)
non_en_mask = (df['language'].astype(str).str.lower() != 'en') if 'language' in df.columns else pd.Series([False]*initial)
kw_mask = pd.Series([False]*initial)
if cols:
    txt = df[cols].fillna('').astype(str)
    haystack = txt.agg(' '.join, axis=1).str.lower()
    kw_mask = haystack.apply(lambda t: any(term in t for term in exclude_terms))

combined = (non_en_mask | kw_mask)
combined_excluded = int(combined.sum())

print(f'Total articles: {initial}')
print(f'Non-English excluded: {non_en} ({non_en/initial:.1%})')
print(f'Keyword excluded: {kw_excluded} ({kw_excluded/initial:.1%})')
print(f'Combined excluded (union): {combined_excluded} ({combined_excluded/initial:.1%})')

# Show top 10 keywords contributing (simple heuristic)
from collections import Counter
cnt = Counter()
if cols:
    for term in exclude_terms:
        cnt[term] = int(haystack.str.contains(term).sum())
    top = cnt.most_common(10)
    print('Top term hits:', top)
