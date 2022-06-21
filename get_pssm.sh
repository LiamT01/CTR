# Make sure you have BLAST installed and the non-redundant (nr) database.
# Please modify the paths in $BLASTDB and /path/to/blast/bin/psiblast before running!

export BLASTDB=/path/to/db/nr

for seq in data/seq/*; do
  /path/to/blast/bin/psiblast -query "${seq}" -num_threads 24 -db nr -num_iterations 3 -out data/out/"${seq:9}".out -out_ascii_pssm data/pssm/"${seq:9}".pssm;
done

