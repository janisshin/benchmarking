#!/bin/bash/

input_file=$1
model_no=$2

#---------------------
j=-1

while IFS= read -r line
do
	if [[ $j -eq 0 ]]
	then
		fl_1=($line)
		fl_2=${#fl_1[@]}
		fl_3=$((fl_2-1))
		j=$((j+1))

	elif [[ $line == "J"*":"*  ]] 
	then
		j=$((j+1))
	fi
	
	if [[ $line == *"; syn"* || $line == *"; deg"* ]] 
	then
		
		l2=`printf '%s' "$line" | cut -d':' -f2`
		n=`echo "$l2" | cut	-c3` 

		replacement="S"$n"_y"
		corrected=${l2//B$n/$replacement}
		
		la=`echo "$corrected" | cut -d';' -f1`
		lb=`echo "$corrected" | cut -d';' -f2`

	  	echo "EX_S$n:$la; E$j*$lb" >> "cbm_ma_$model_no.txt"
  	#elif [[ $line == "kf$((j-fl_3))"* ]]
	#then
	#	echo $line >> "cbm_ma_$model_no.txt"
	#	for i in 1 2 3
	#	do 
	#		r_n=`seq 0 .01 1 | shuf | head -n1`
	#		echo "kf$((fl_3+$i)) = $r_n" >> "cbm_ma_$model_no.txt"
	#	done
	elif [[ $line == "B"*"="* ]]
	then	
		l1=`printf '%s' "$line" | cut -d'=' -f1`
		l2=`printf '%s' "$line" | cut -d'=' -f2`
		n=`echo "$l1" | cut	-c2` 

		replacement="S"$n"_y"
		corrected=${line//B$n/$replacement}

	  	echo "$corrected" >> "cbm_ma_$model_no.txt"



	else
		echo $line >> "cbm_ma_$model_no.txt"
	fi

done < "$input_file"

for i in 0 1 2 
do 
	r_n=`seq 0 .01 1 | shuf | head -n1`
	echo "E$((fl_3+$i)) = $r_n" >> "cbm_ma_$model_no.txt"
done

