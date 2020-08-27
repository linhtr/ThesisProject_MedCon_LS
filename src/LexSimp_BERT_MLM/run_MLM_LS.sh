export Input_DIR=/Users/linh/Documents/GitHub/Thesis_Project/LexSimp_CHV_rule-based/CHV_substitution/output/
export Model_DIR=/Users/linh/Documents/GitHub/Thesis_Project/LexSimp_BERT_MLM/
export Output_DIR=/Users/linh/Documents/GitHub/Thesis_Project/LexSimp_BERT_MLM/output/

python MLM_LS.py \
  --do_eval \
  --do_lower_case \
  --num_selections 10 \
  --eval_dir $Input_DIR/subtitles/CHV_singlew_source_subtitles_final.csv \
  --bert_model emilyalsentzer/Bio_Discharge_Summary_BERT \
  --cache_dir $Model_DIR/transformers_pretrained_models/ \
  --max_seq_length 250 \
  --word_embeddings $Model_DIR/word_embeddings/fastText/crawl-300d-2M-subword.vec \
  --word_frequency $Model_DIR/word_frequency/counter_Tokens.p \
  --stopwords $Model_DIR/MySQL_MyISAM_stopwords.txt \
  --output_path $Output_DIR/subtitles/MLM_sw_source_subtitles_substitution_clinicalbert_disch.csv

#  --bert_model bert-large-cased-whole-word-masking \
