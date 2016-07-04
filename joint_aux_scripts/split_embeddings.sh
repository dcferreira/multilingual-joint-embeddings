input_file=$1
output_prefix=$2

for lang in en ar de es fr it nl pl pb ro ru tr
do
    echo "Creating ${output_prefix}.${lang}..."
    grep "_${lang}\s" ${input_file} | sed "s/_${lang}\(\s\)/\1/" \
        > ${output_prefix}.$lang
done
