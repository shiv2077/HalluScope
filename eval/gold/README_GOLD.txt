GOLD GRADER-CALIBRATION WORKFLOW
================================
1. (done) Template created: gold_questions.json + gold_annotatorA.csv +
   gold_annotatorB.csv. model_answer is intentionally BLANK.

2. Generate the model's REAL answers for these exact questions:
     python3 eval/run_benchmark.py --mode local --eval-deterministic \
         --dataset-file eval/gold/gold_questions.json \
         --out eval/gold/gold_results.csv
   (use --mode api for the API model; needs OPENAI_API_KEY)

3. Merge those real answers into BOTH annotator CSVs:
     python3 eval/make_gold_template.py --fill-from eval/gold/gold_results.csv

4. Hand-label: in gold_annotatorA.csv and gold_annotatorB.csv set human_label
   to 'correct' or 'incorrect' for every row, judging whether model_answer is
   equivalent to reference_answer. Two people label independently.

5. Calibrate against the human labels:
     python3 eval/grader_calibration.py \
         --gold eval/gold/gold_annotatorA.csv \
         --gold-b eval/gold/gold_annotatorB.csv

6. Review inter-annotator kappa, then set config/grader.json -> sbert_threshold
   to the recommended value (the command is printed by step 5).
