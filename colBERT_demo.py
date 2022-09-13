
import os
import sys
sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

### loading data
dataroot = '../data/'
column_collection = os.path.join(dataroot, 'columns_withdb.tsv')
location_collection = os.path.join(dataroot, 'locations_withdb.tsv')
column_collection = Collection(path=column_collection)
location_collection = Collection(path=location_collection)
print(f'Loaded {len(column_collection)} columns and {len(location_collection):,} locations')

### loading checkpoints
nbits = 2
doc_maxlen = 300   

checkpoint = 'downloads/colbertv2.0'
column_index_name = f'datacommons.individual_columns.{nbits}bits'
location_index_name = f'datacommons.individual_locations.{nbits}bits'

### loading indexer
with Run().context(RunConfig(nranks=1, experiment='notebook')):  
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

    column_indexer = Indexer(checkpoint=checkpoint, config=config)
    column_indexer.index(name=column_index_name, collection=column_collection, overwrite=True)

    location_indexer = Indexer(checkpoint=checkpoint, config=config)
    location_indexer.index(name=location_index_name, collection=location_collection, overwrite=True)

### loading searcher
with Run().context(RunConfig(experiment='notebook')):
    column_searcher = Searcher(index=column_index_name)
    location_searcher = Searcher(index=location_index_name)

### fun to get subset
def get_cols_locs(category, query):
    column_results = column_searcher.search(query, k=10)
    location_results = location_searcher.search(query, k=10)

    columns = []
    for passage_id, passage_rank, passage_score in zip(*column_results):
        if category in column_searcher.collection[passage_id]:
            columns.append(column_searcher.collection[passage_id].replace(category, ''))

    locations = []
    for passage_id, passage_rank, passage_score in zip(*location_results):
        if category in location_searcher.collection[passage_id]:
            locations.append(location_searcher.collection[passage_id].replace(category, ''))

    return columns, locations

### sample run
query="total farm are in Dane County"
category="AGRICULTURE"

cols, locs = get_cols_locs(category, query)
print(cols)
print(locs)


