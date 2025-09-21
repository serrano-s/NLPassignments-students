wget https://raw.githubusercontent.com/serrano-s/NLPassignments-students/refs/heads/main/assignments/WordEmbeddings/wordvec_tests.py
glovepartone="data/embeddings/glove.6B/glove.6B.50d.part1.txt"
gloveparttwo="data/embeddings/glove.6B/glove.6B.50d.part2.txt"
overallglove="data/embeddings/glove.6B/glove.6B.50d.txt"
cat "$gloveparttwo" >> "$glovepartone"
mv "$glovepartone" "$overallglove"
rm "$gloveparttwo"
