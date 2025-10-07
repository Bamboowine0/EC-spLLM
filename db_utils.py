import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import random
import logging
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_path = "your model path"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path, device="cuda")

def dis_func(embedding_function="cosine"):
    """
    Return the configuration dictionary according to the specified embedding distance function type

    Args:
        embedding_function (str): Distance function type, optional values are 'l2', 'ip', 'cosine'

    Returns:
        dict: Dictionary containing distance function configuration
    """
    distance_map = {
        "l2": {"hnsw:space": "l2"},
        "ip": {"hnsw:space": "ip"},
        "cosine": {"hnsw:space": "cosine"}
    }
    return distance_map.get(embedding_function, {"hnsw:space": "cosine"})

def create_db(path, kgname, dis_function="cosine"):
    """
    Create a new knowledge graph database

    Args:
        path (str): Database storage path
        kgname (str): Knowledge graph name
        dis_function (str): Distance function type

    Returns:
        tuple: (client, collection) Client and collection object
    """
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(
            name=kgname,
            metadata=dis_func(dis_function),
            embedding_function=sentence_transformer_ef
        )
        logger.info(f"Knowledge graph {kgname} created successfully")
        return client, collection
    except Exception as e:
        logger.error(f"Failed to create knowledge graph: {e}")
        raise

def open_db(path, kgname):
    """
    Open an existing knowledge graph database

    Args:
        path (str): Database storage path
        kgname (str): Knowledge graph name

    Returns:
        tuple: (client, collection) Client and collection object
    """
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(name=kgname, embedding_function=sentence_transformer_ef)
        logger.info(f"Knowledge graph {kgname} opened successfully")
        return client, collection
    except Exception as e:
        logger.error(f"Failed to open knowledge graph: {e}")
        raise

def init_db(db_path, kg_name, dis_function="cosine"):
    """
    Initialize the knowledge graph database, create if not exists, open if exists

    Args:
        db_path (str): Database storage path
        kg_name (str): Knowledge graph name
        dis_function (str): Distance function type

    Returns:
        tuple: (db_path, collection) Database path and collection object
    """
    try:
        _, collection = open_db(db_path, kg_name)
        logger.info(f"Knowledge graph loaded: {kg_name}")
    except Exception:
        logger.info(f"Knowledge graph {kg_name} does not exist, creating...")
        _, collection = create_db(db_path, kg_name, dis_function)
    return db_path, collection

def db_count(path, kgname):
    """
    Get the number of triples in the knowledge graph

    Args:
        path (str): Database storage path
        kgname (str): Knowledge graph name

    Returns:
        int: Number of triples
    """
    _, collection = open_db(path, kgname)
    count = collection.count()
    logger.info(f"Knowledge graph {kgname} contains {count} triples")
    return count

def find_max_index_smaller_than_n(lst, n):
    """
    Find the maximum index in the list where the value is less than n

    Args:
        lst (list): List of numbers
        n (float): Threshold

    Returns:
        int: The maximum index that meets the condition, or len(lst)-1 if not found
    """
    for i, num in enumerate(lst):
        if num > n:
            return i - 1
    return len(lst) - 1

def query_db_search(question, query_content, collection, width, dist, sim_threshold, sim_top, model):
    """
    Query triples related to the question in the knowledge graph

    Args:
        question (str): Query question
        query_content (str): Query content
        collection (Collection): ChromaDB collection object
        width (int): Maximum number of results returned
        dist (float): Distance threshold
        sim_threshold (float): Similarity threshold, used to filter out low similarity triples
        sim_top (int): Return the top N triples with the highest similarity
        model (SentenceTransformer): Model for calculating text embeddings

    Returns:
        list: List of triples that meet the condition, or None if not found
    """
    if collection.count() == 0:
        return None

    res = collection.query(
        query_texts=query_content,
        n_results=width,
        include=["documents", "metadatas", "distances"]
    )

    if not res['metadatas'] or len(res['metadatas'][0]) == 0:
        return None

    triple = res['metadatas'][0]
    documents = res['documents'][0]
    distances = res['distances'][0]

    candidates_list = []
    triple_list = []
    k = find_max_index_smaller_than_n(distances, dist)
    if k == -1:
        return None
    else:
        candidates_list = documents[:k+1]
        triple_list = triple[:k+1]

    print("Candidate list")
    print(candidates_list)
    print("--------------------------------")

    question_emb = model.encode(question, convert_to_tensor=True)
    knowledge_embs = model.encode(candidates_list, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(question_emb, knowledge_embs)

    filtered = [(triple_list[i], sims[0, i].item()) for i in range(len(triple_list)) if sims[0, i].item() > sim_threshold]
    print("Similarity list")
    print(filtered)
    print("--------------------------------")
    filtered.sort(key=lambda x: x[1], reverse=True)
    top3 = [item[0] for item in filtered[:sim_top]]

    return top3 if top3 else None

def find_entity_in_kg(entity, position, collection):
    """
    Find triples in the knowledge graph that contain the specified entity at the given position

    Args:
        entity (str): Entity to search for
        position (str): Position of the entity in the triple (head/tail)
        collection (Collection): ChromaDB collection object

    Returns:
        tuple: (exists, result dictionary)
    """
    dict0 = collection.get(where={position: entity})
    return len(dict0["ids"]) > 0, dict0

def find_triple_in_kg(triple, collection):
    """
    Find a specific triple in the knowledge graph

    Args:
        triple (dict): Triple dictionary containing head and tail
        collection (Collection): ChromaDB collection object

    Returns:
        tuple: (exists, triple ID)
    """
    head_exist, head_triple = find_entity_in_kg(triple["head"], "head", collection)
    tail_exist, tail_triple = find_entity_in_kg(triple["tail"], "tail", collection)

    if head_exist and tail_exist:
        for i in head_triple["ids"]:
            if i in tail_triple["ids"]:
                return True, i
        return False, ''
    else:
        return False, ''

def kg_add_triple_emb(path, kgname, triple, qa):
    """
    Add a new triple (with embedding) to the knowledge graph

    Args:
        path (str): Database storage path
        kgname (str): Knowledge graph name
        triple (dict): Triple to add
        qa (str): QA content

    Returns:
        tuple: (path, kgname, triple, collection)
    """
    client, collection = open_db(path, kgname)
    triple_in_kg, id = find_triple_in_kg(triple, collection)

    if triple_in_kg:
        logger.info(f"Triple {triple} already exists in knowledge graph {kgname}")
    else:
        count = collection.count()
        collection.add(
            documents=' '.join(list(triple.values())),
            metadatas=triple,
            ids=f"triple{count+1}"
        )
        logger.info(f"Triple {triple} added to knowledge graph {kgname}")

    return path, kgname, triple, collection

def kg_add_triple(triple, collection):
    """
    Add a new triple to the knowledge graph

    Args:
        triple (dict): Triple to add
        collection (Collection): ChromaDB collection object

    Returns:
        bool: Whether the addition was successful
    """
    try:
        triple_in_kg, _ = find_triple_in_kg(triple, collection)

        if triple_in_kg:
            logger.info(f"Triple {triple} already exists in the knowledge graph")
            return True

        count = collection.count()
        collection.add(
            documents=''.join(list(triple.values())),
            metadatas=triple,
            ids=f"triple{count+1}"
        )
        logger.info(f"Triple {triple} added to the knowledge graph successfully")
        return True
    except Exception as e:
        logger.error(f"Error adding triple: {e}")
        return False
