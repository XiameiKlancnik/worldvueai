import pandas as pd
from worldvue.data.loaders import load_articles

current = pd.read_parquet('articles_with_embeddings.parquet')
print('older parquet rows:', len(current))

articles = load_articles('all_articles.parquet')
with_embeddings = sum(1 for art in articles if art.embedding)
print('loaded articles:', len(articles))
print('articles with cached embeddings:', with_embeddings)
