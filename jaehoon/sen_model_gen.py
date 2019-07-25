import sentencepiece as spm


class Sentence_model_gen():

    def __init__(self, prefix_en, prefix_de, config):

        self.en_model_gen(prefix_en)
        self.de_model_gen(prefix_de)
        self.input_file = config.input_file_dir
        self.output_file = config.output_file_dir


    def en_model_gen(self, prefix_en):

        templates = '--input={} --model_prefix={} --vocab_size={} --control_symbols=[CLS],[SEP]\
                    --user_defined_symbols=[MASK] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'
        vocab_size = 5000
        prefix = prefix_en
        cmd = templates.format(self.input_file, prefix, vocab_size)
        spm.SentencePieceTrainer.Train(cmd)


    def de_model_gen(self, prefix_de):

        templates = '--input={} --model_prefix={} --vocab_size={} --control_symbols=[CLS],[SEP]\
                    --user_defined_symbols=[MASK] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'
        vocab_size = 5000
        prefix = prefix_de
        cmd = templates.format(self.output_file, prefix, vocab_size)
        spm.SentencePieceTrainer.Train(cmd)