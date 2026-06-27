from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(self, file_name: str, chunk_size: int = 300, overlap: int = 50, dimensions: int = 384, truncate_table: bool = False):
        if chunk_size < 10:
            raise ValueError('chunk_size must be at least 10')
        if overlap < 0:
            raise ValueError('overlap must be at least 0')
        if overlap >= chunk_size:
            raise ValueError('overlap should be lower than chunkSize')

        if truncate_table:
            self._truncate_table()

        print('Preparing text chunks ...')
        with open(file_name, "r", encoding='utf-8') as file:
            text = file.read()

        text_chunks = chunk_text(text, chunk_size, overlap)
        print('Getting embeddings from chunks ...')
        indexed_embeddings = self.embeddings_client.get_embeddings(text_chunks, dimensions)

        print('Inserting records to the vectors table ...')
        for idx, embedding in indexed_embeddings.items():
            self._save_chunk(file_name, text_chunks[idx], indexed_embeddings[idx])

    def _truncate_table(self,):
        print('Truncating vectors table ...')
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('TRUNCATE TABLE vectors')
                conn.commit()

    def _save_chunk(self, file_name, chunk, embedding_list):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO vectors(document_name, text, embedding) VALUES (%s, %s, %s::vector)",(file_name, chunk, embedding_list))
                conn.commit()

    def search(self, search_mode, query: str, top_k: int = 4, score: int = 0.5, dimensions: int = 384):
        if top_k < 1:
            raise ValueError('top_k must be at least 1')
        if score < 0 or score > 1:
            raise ValueError('score_threshold must be in [0.0..., 0.99...] range')

        query_embeddings = self.embeddings_client.get_embeddings(query, dimensions)
        embedding_list = query_embeddings[0]

        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        sign = None
        max_distance = None
        if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
            max_distance = float('inf') if score == 0 else (1.0 / score) - 1.0
            sign = '<->'
        elif search_mode == 'cosine':
            max_distance = 1.0 - score
            sign = '<=>'

        sql_query = f"""
        SELECT text, embedding {sign}  %s::vector AS distance
        FROM vectors
        WHERE embedding {sign}  %s::vector <= %s
        ORDER BY distance
        LIMIT %s;
        """

        text_chunks = []

        cur.execute(sql_query, (embedding_list, embedding_list, max_distance, top_k))
        results = cur.fetchall()
        for row in results:
            text_chunks.append(row['text'])

        return '\n\n'.join(text_chunks)

