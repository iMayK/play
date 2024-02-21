def _hash_schema(
    dataset: Dataset,
    is_eval
) -> Dataset:
    chars = string.ascii_letters 
    digits = string.digits
    oID = lambda oID_length: random.choices(chars, k=1)[0]  + "".join(random.choices(chars + digits, k=oID_length))
    oID_length = 4
    oID_separator = " "

    def get_hash(entity, entity2hash, hash_codes):
        if entity in entity2hash:
            return entity2hash[entity]
        while 1:
            hash_code = oID(oID_length)
            if hash_code not in hash_codes:
                hash_codes.add(hash_code)
                entity2hash[entity] = hash_code
                return hash_code

    def _add_prefix(example, dbid_cache):
        db_id = example["db_id"]
        if (not is_eval) or db_id not in dbid_cache:
            hash_codes = set()
            entity2hash = {}

            example["db_unhashed_table_names"] = [entity.lower() for entity in example["db_table_names"]] 
            example["db_unhashed_column_names"] = {
                    "table_id": example["db_column_names"]["table_id"],
                    "column_name": [entity.lower() for entity in example["db_column_names"]["column_name"]]
            }

            example["db_table_names"] = [f"{get_hash(entity.lower(), entity2hash, hash_codes)} {entity.lower()}" for entity in example["db_table_names"]]
            example["db_column_names"] = {
                    "table_id": example["db_column_names"]["table_id"],
                    "column_name": [f"{get_hash(entity.lower(), entity2hash, hash_codes)} {entity.lower()}" if entity!='*' else entity for entity in example["db_column_names"]["column_name"]]
            }

            dbid_cache[db_id] = {
                    "db_table_names": example["db_table_names"],
                    "db_column_names": example["db_column_names"],
                    "entity2hash": entity2hash
            }
        else:
            example["db_unhashed_table_names"] = deepcopy(example["db_table_names"])
            example["db_unhashed_column_names"] = deepcopy(example["db_column_names"])
            example["db_table_names"] = dbid_cache[db_id]["db_table_names"]
            example["db_column_names"] = dbid_cache[db_id]["db_column_names"]
            entity2hash = dbid_cache[db_id]["entity2hash"]

        entities_sorted = sorted(entity2hash.keys(), key=lambda x: len(x), reverse=True)
        query = normalize(example["query"])
        orig_query = query 
        for entity in entities_sorted:
            query = query.replace(entity, entity2hash[entity])
        example["query"] = query
        example["orig_query"] = orig_query
        example["entity2hash"] = " | ".join([f"{k} {v}" for k, v in entity2hash.items()])

        return example
    
    dbid_cache = {}
    dataset = dataset.map(
            _add_prefix,
            fn_kwargs={"dbid_cache": dbid_cache},
            batched=False,
            load_from_cache_file=False,
    )

    if is_eval:
        dataset = create_new_dbs(dbid_cache, dataset)
    
    return dataset
