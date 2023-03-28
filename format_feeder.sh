#!/bin/bash/

mapfile -t models < bd_3sp.txt

file_loc=`pwd`

for m in "${models[@]}"
do
	m=`echo "${m//[$'\t\r\n ']}"`
	input_file="${file_loc}/models/mass_action/antimony/mass_action_$m.ant"
	
	bash formatting_for_cobra.sh $input_file $m

done
