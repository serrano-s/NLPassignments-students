wget $(python data_path_to_download_url.py "assignments/NeuralNetworks/WithSentimentAnalysis/nn_tests.py")
wget $(python data_path_to_download_url.py "assignments/NeuralNetworks/WithSentimentAnalysis/siqa.py")
wget $(python data_path_to_download_url.py "assignments/NeuralNetworks/WithSentimentAnalysis/assemble_socialiqa_train_embeds.py")
glovepartone="data/embeddings/glove.6B/glove.6B.50d.part1.txt"
gloveparttwo="data/embeddings/glove.6B/glove.6B.50d.part2.txt"
overallglove="data/embeddings/glove.6B/glove.6B.50d.txt"
cat "$gloveparttwo" >> "$glovepartone"
mv "$glovepartone" "$overallglove"
rm "$gloveparttwo"
python assemble_socialiqa_train_embeds.py
rm assemble_socialiqa_train_embeds.py
