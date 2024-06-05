import polars as pl
import drain3 as dr
import hashlib
import os

# Drain.ini default regexes
# No lookahead or lookbedinde so reimplemented with capture groups. Still problem with overlaps See
# https://docs.rs/regex/latest/regex/
# https://stackoverflow.com/questions/57497045/how-to-get-overlapping-regex-captures-in-rust
# Orig:     BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590. blk_-1727475099218615100
# After 1st BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_0001_m_<NUM>_0/part-<NUM>. blk_<SEQ>'
# After 2nd BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_<NUM>_m_<NUM>_<NUM>/part-<NUM>. blk_<SEQ>'
masking_patterns_drain = [
    ("${start}<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9a-f]{6,} ?){3,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9A-F]{4} ?){4,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]


class EventLogEnhancer:
    def __init__(self, df):
        self.df = df

    # Helper function to check if all prerequisites exist
    def _prerequisites_exist(self, prerequisites):
        return all([col in self.df.columns for col in prerequisites])

    # Helper function to handle prerequisite check and raise exception if missing
    def _handle_prerequisites(self, prerequisites):
        if not self._prerequisites_exist(prerequisites):
            raise ValueError(f"Missing prerequisites for enrichment: {', '.join(prerequisites)}")

    # Function-based enricher to split messages into words
    def words(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_words" not in self.df.columns:
            self.df = self.df.with_columns(pl.col(column).str.split(by=" ").alias("e_words"))
            self.df = self.df.with_columns(
                e_words_len = pl.col("e_words").list.lengths(),
            )
        else:
            print("e_words already found")
        return self.df

    # Function-based enricher to extract alphanumeric tokens from messages
    def alphanumerics(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_alphanumerics" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).str.extract_all(r"[a-zA-Z\d]+").alias("e_alphanumerics")
            )
            self.df = self.df.with_columns(
                e_alphanumerics_len = pl.col("e_alphanumerics").list.lengths(),
            )
        return self.df

    # Function-based enricher to create trigrams from messages
    # Trigrams enrichment is slow 1M lines in 40s.
    # Trigram flag to be removed after this is fixed.
    # https://github.com/pola-rs/polars/issues/10833
    # https://github.com/pola-rs/polars/issues/10890
    def trigrams(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_trigrams" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).map_elements(
                    lambda mes: self._create_cngram(message=mes, ngram=3)).alias("e_trigrams")
            )
            self.df = self.df.with_columns(
                e_trigrams_len = pl.col("e_trigrams").list.lengths()
            )
        return self.df

    def _create_cngram(self, message, ngram=3):
        if ngram <= 0:
            return []
        return [message[i:i + ngram] for i in range(len(message) - ngram + 1)]

    # Enrich with drain parsing results
    def parse_drain(self, drain_masking=False, reparse=False):
        self._handle_prerequisites(["m_message"])
        if reparse or "e_event_id" not in self.df.columns:
            # Drain returns dict
            # {'change_type': 'none',
            # 'cluster_id': 1,
            # 'cluster_size': 2,
            # 'template_mined': 'session closed for user root',
            # 'cluster_count': 1}
            # we store template for later use.

            # We might have multiline log message, i.e. log_message + stack trace.
            # Use only first line of log message for parsing
            current_script_path = os.path.abspath(__file__)
            current_script_directory = os.path.dirname(current_script_path)
            drain3_ini_location =  os.path.join(current_script_directory, '../parsers/drain3/')
            if drain_masking:
                dr.template_miner.config_filename = os.path.join(drain3_ini_location,'drain3.ini') #TODO fix the path relative
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    message_trimmed=pl.col("m_message").str.split("\n").list.first()
                )
                self.df = self.df.with_columns(
                    drain=pl.col("message_trimmed").map_elements(lambda x: self.tm.add_log_message(x)))
            else:
                if "e_message_normalized" not in self.df.columns:
                    self.normalize()
                dr.template_miner.config_filename =os.path.join(drain3_ini_location, 'drain3_no_masking.ini') #drain3_no_masking.ini'  #TODO fix the path relative
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    drain=pl.col("e_message_normalized").map_elements(lambda x: self.tm.add_log_message(x)))

            self.df = self.df.with_columns(
                e_event_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8),
                # extra thing ensure we e1 e2 instead of 1 2
                e_template=pl.col("drain").struct.field("template_mined"))
            self.df = self.df.drop("drain")  # Drop the dictionary produced by drain. Event_id and template are the most important.
            # tm.drain.print_tree()
        return self.df

    def length(self, column="m_message"):
        self._handle_prerequisites(["m_message"])
        if "e_chars_len" not in self.df.columns:
            self.df = self.df.with_columns(
                e_chars_len=pl.col(column).str.n_chars(),
                e_lines_len=pl.col(column).str.count_matches(r"(\n|\r|\r\n)"),
                e_event_id_len = 1 #Messages are always one event. Added to simplify code later on. 
 
            )
        return self.df

    def normalize(self, regexs=masking_patterns_drain, to_lower=False, twice=True):

        # base_code = 'self.df = self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'
        base_code = 'self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'

        if to_lower:
            base_code += '.str.to_lowercase()'

        # Generate the replace_all chain
        # TODO We need to duplicate everything otherwise we get only every other replacement in 
        #"Folder_0012_2323_2324" -> After first replacement we get Folder Folder_<NUM>_2323_<NUM>
        #After second replacement we get  Folder_<NUM>_<NUM>_<NUM>. This is ugly but due to Crate limitations
        # https://docs.rs/regex/latest/regex/
        # https://stackoverflow.com/questions/57497045/how-to-get-overlapping-regex-captures-in-rust
        # Orig:     BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590. blk_-1727475099218615100
        # After 1st BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_0001_m_<NUM>_0/part-<NUM>. blk_<SEQ>'
        # After 2nd BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_<NUM>_m_<NUM>_<NUM>/part-<NUM>. blk_<SEQ>'
        #Longer explanation Overlapping Matches: The regex crate does not find overlapping matches by default. If your text has numbers that are immediately adjacent to each other with only a non-alphanumeric separator (which is consumed by the start or end group), the regex engine won't match the second number because the separator is already consumed by the first match.
        for key, pattern in regexs:
            replace_code = f'.str.replace_all(r"{pattern}", "{key}")'
            base_code += replace_code
            if twice:
                base_code += replace_code

        base_code += ')'
        self.df = eval(base_code)
        return self.df
        # print (base_code)
        # return base_code


class SequenceEnhancer:
    def __init__(self, df, df_seq):
        self.df = df
        self.df_seq = df_seq

    def start_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').min().alias('start_time'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def end_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').max().alias('end_time'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def seq_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.count().alias('seq_len'))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        self.df_seq = self.df_seq.with_columns(self.df_seq['seq_len'].alias('e_event_id_len'))

        return self.df_seq

    def events(self, event_col = "e_event_id"):
        # Aggregate event ids into a list for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.col(event_col).alias(event_col))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq


    def tokens(self, token="e_words"):
        #df_temp = self.df.group_by('seq_id').agg(pl.col(token).flatten().alias(token))
        #Same as above but the above crashes due to out of memory problems. We might need this fix also in other rows
        df_temp = self.df.select("seq_id", token).explode(token).group_by('seq_id').agg(pl.col(token))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        #lengths
        df_temp = self.df.group_by('seq_id').agg(pl.col(token+"_len").sum().alias(token+"_len"))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        return self.df_seq
    
    def duration(self):
        # Calculate the sequence duration for each seq_id as the difference between max and min timestamps
        df_temp = self.df.group_by('seq_id').agg(
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).alias('duration'),
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).dt.seconds().alias('duration_sec')
        )
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def eve_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(
            eve_len_max=pl.col('e_chars_len').max(),
            eve_len_min=pl.col('e_chars_len').min(),
            eve_len_avg=pl.col('e_chars_len').mean(),
            eve_len_med=pl.col('e_chars_len').median(),
            eve_len_over1=(pl.col('e_chars_len') > 1).sum()
        )
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq     
    